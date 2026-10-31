import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def _exact_ratio(h):
    """
    Compute the exact ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = fftconvolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _is_sidon_set(indices):
    """Check if a set of indices forms a B2 (Sidon) set: all pairwise sums are distinct."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _generate_sidon_candidates():
    """
    Generate a list of candidate Sidon sets of various sizes.
    """
    candidates = []
    
    # Known good base structures
    base_sets = [
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30, 35, 40],
        [0, 1, 4, 9, 15, 21, 27, 33, 39, 44, 48],
        [0, 2, 5, 11, 14, 20, 26, 32, 38, 43, 47],
        [0, 1, 3, 7, 12, 18, 25, 31, 38, 44, 49],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 30],
        [0, 1, 4, 9, 15, 21, 27, 33, 39],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 9, 15, 21],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 9, 15],
        [0, 1, 4, 10],
        [0, 1, 4, 9],
        [0, 1, 4],
        [0, 1, 3],
    ]
    
    for s in base_sets:
        if all(0 <= x < N for x in s) and _is_sidon_set(s):
            candidates.append(list(s))
    
    # Random generation of Sidon sets
    np.random.seed(123)
    existing = [frozenset(c) for c in candidates]
    for _ in range(3000):
        k = np.random.randint(4, 13)
        indices = sorted(np.random.choice(N, k, replace=False).tolist())
        if _is_sidon_set(indices):
            fs = frozenset(indices)
            if fs not in existing:
                candidates.append(indices)
                existing.append(fs)
                if len(candidates) > 80:
                    break
    return candidates

def _optimize_weights(support, deadline):
    """
    Optimize weights on a fixed support set to maximize the ratio.
    Uses FFT-based gradient for the smooth max surrogate.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    # Initial weights: equal
    w_init = np.ones(k) / k
    h_init = np.zeros(N)
    h_init[support] = w_init
    best_score = _exact_ratio(h_init)
    best_w = w_init.copy()
    
    # Try multiple betas for the smooth max
    # Higher beta -> sharper max -> better approximation of true objective
    for beta in [50.0, 100.0, 200.0, 400.0]:
        if time.time() > deadline - 2:
            break
            
        def _neg_ratio_smooth(params):
            # params are log weights to ensure positivity
            w_local = np.exp(params)
            s = np.sum(w_local)
            if s < 1e-15:
                return 0.0, np.zeros(k)
            w_local = w_local / s
            
            h_local = np.zeros(N)
            h_local[support] = w_local
            
            l2_sq = np.sum(h_local**2)
            conv = fftconvolve(h_local, h_local, mode='full')
            
            # Smooth max calculation
            m = np.max(conv)
            if m < 0: m = 0
            shifted = conv - m
            shifted = np.clip(shifted, -500, 500)
            log_sum_exp = m + np.log(np.sum(np.exp(beta * shifted)))
            smooth_max = log_sum_exp / beta
            
            if smooth_max < 1e-15:
                return 0.0, np.zeros(k)
                
            # Objective: -l2_sq / smooth_max
            obj_val = -l2_sq / smooth_max
            
            # Gradient calculation
            # We need d/dw ( -l2_sq / smooth_max )
            # Let f = l2_sq, g = smooth_max
            # d(f/g)/dw = (f' g - f g') / g^2
            # d/dw (-f/g) = (f g' - f' g) / g^2
            
            # 1. Gradient of l2_sq w.r.t w
            # l2_sq = sum w_i^2
            # d l2_sq / d w_i = 2 w_i
            grad_l2 = 2.0 * w_local
            
            # 2. Gradient of smooth_max w.r.t w using FFT
            # smooth_max = (1/beta) log(sum_k exp(beta conv_k))
            # d smooth_max / d w_j = sum_k p_k * d conv_k / d w_j
            # where p_k = exp(beta conv_k) / sum_exp
            # d conv_k / d w_j = 2 * h_{k-j} (circular, but we use linear conv so need care)
            # Actually for linear convolution: conv[k] = sum_m h[m] * h[k-m]
            # d conv[k] / d h[j] = 2 * h[k-j] if k-j in [0,N-1], else 0 (for symmetric h)
            # Wait, conv[k] = sum_{m=0}^{N-1} h[m] h[k-m] where h is defined on [0,N-1]
            # d conv[k] / d h[j] = h[k-j] + h[j-(k-N+1)] ... this gets complicated
            # 
            # Better approach: Use the fact that for real h, the autocorrelation is:
            # R[k] = sum_m h[m] h[m+k] for k >= 0
            # R[-k] = R[k]
            # In full mode, conv has length 2N-1, with conv[k] = R[k-(N-1)]
            # So d conv[k] / d h[j] = 2 * h[k - (N-1) - j + (N-1)] ... 
            # 
            # Let me think differently. conv_full[k] = sum_m h[m] * h[k-m] for valid m
            # d conv_full[k] / d h[j] = h[k-j] + (if j != k-j) h[k-j] = 2*h[k-j] when k-j in [0,N-1]
            # But actually conv_full[k] = sum_{m=max(0,k-N+1)}^{min(N-1,k)} h[m] h[k-m]
            # d conv_full[k] / d h[j] = h[k-j] (if 0<=k-j<N) + h[j] (if 0<=k-j<N and j != k-j)
            # = 2*h[k-j] if 0<=k-j<N and j != k-j-j... 
            # 
            # Simpler: just compute p[k] * 2*h[k-j] summed over k where k-j is in [0,N-1]
            # grad_smooth[j] = 2 * sum_{k: 0<=k-j<N} p[k] * h[k-j]
            # Let t = k-j, then k = t+j, t in [0,N-1]
            # grad_smooth[j] = 2 * sum_{t=0}^{N-1} p[t+j] * h[t]
            # 
            # This is a cross-correlation! We can compute it with FFT.
            # Let P be the FFT of p (padded to appropriate size)
            # Let H be the FFT of h
            # The cross-correlation sum_t p[t+j] h[t] = ifft(FFT(p) * conj(FFT(h)))[j]
            # But we need sum_t p[t+j] h[t], which is (p * h_flipped)[j + N - 1] or similar
            
            exp_shifted = np.exp(beta * shifted)
            sum_exp = np.sum(exp_shifted)
            p = exp_shifted / sum_exp  # Length 2N-1
            
            # Compute grad_smooth using FFT-based cross-correlation
            # We want for each j in support: grad_smooth[j] = 2 * sum_{t=0}^{N-1} p[t+j] * h[t]
            # This is the cross-correlation of p and h, evaluated at lag j
            # cross_corr[j] = sum_t p[t+j] * h[t]
            # Using FFT: cross_corr = ifft(FFT(p_padded) * conj(FFT(h_padded)))
            # where we pad to length >= (2N-1) + N - 1 = 3N-2, use 4N for safety
            
            fft_size = 1
            while fft_size < 3 * N:
                fft_size *= 2
            
            p_padded = np.zeros(fft_size)
            p_padded[:2*N-1] = p
            
            h_padded = np.zeros(fft_size)
            h_padded[:N] = h_local
            
            P_fft = np.fft.rfft(p_padded)
            H_fft = np.fft.rfft(h_padded)
            
            # Cross-correlation: sum_t p[t+j] h[t] = ifft(FFT(p) * conj(FFT(h)))
            # This gives cross_corr[j] for j = 0, 1, ..., fft_size-1
            cross_corr = np.fft.irfft(P_fft * np.conj(H_fft))
            
            # cross_corr[j] = sum_{t=0}^{fft_size-1} p[t+j mod fft_size] * h[t]
            # For j in [0, N-1], this should give us sum_t p[t+j] * h[t] 
            # (since p is zero for t+j >= 2N-1 and h is zero for t >= N)
            
            grad_smooth = np.zeros(k)
            for idx_j, j in enumerate(support):
                grad_smooth[idx_j] = 2.0 * cross_corr[j]
            
            # Combine gradients
            # d obj / d w_j = (f * grad_g_j - g * grad_f_j) / g^2
            # f = l2_sq, g = smooth_max
            # grad_obj = (l2_sq * grad_smooth - smooth_max * grad_l2) / smooth_max^2
            
            grad_w = (l2_sq * grad_smooth - smooth_max * grad_l2) / (smooth_max ** 2)
            
            # Chain rule: d/d(params) = d/dw * dw/d(params)
            # w = exp(params) / sum(exp(params))
            # dw_i / dparams_j = w_i * (delta_ij - w_j)
            # grad_params_j = sum_i grad_w_i * w_i * (delta_ij - w_j)
            # = w_j * grad_w_j - w_j * (sum_i grad_w_i w_i)
            
            dot_grad_w_w = np.dot(grad_w, w_local)
            grad_params = w_local * (grad_w - dot_grad_w_w)
            
            return obj_val, grad_params

        try:
            # Use L-BFGS-B with analytical gradient
            res = minimize(
                _neg_ratio_smooth,
                np.log(w_init + 1e-10),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            w_res = np.exp(res.x)
            s = np.sum(w_res)
            if s > 1e-15:
                w_res = w_res / s
            else:
                w_res = w_init
            
            h_res = np.zeros(N)
            h_res[support] = w_res
            
            score = _exact_ratio(h_res)
            if score > best_score:
                best_score = score
                best_w = w_res.copy()
                w_init = w_res.copy()
        except Exception:
            pass
            
    # Final polish: Coordinate descent on the exact objective
    h_best = np.zeros(N)
    h_best[support] = best_w
    
    # Simple local search: try small perturbations
    for _ in range(30):
        if time.time() > deadline - 1:
            break
        h_curr = h_best.copy()
        curr_score = _exact_ratio(h_curr)
        improved = False
        
        for idx, j in enumerate(support):
            # Try increasing or decreasing w_j slightly
            for delta in [0.01, -0.01, 0.05, -0.05]:
                h_test = h_curr.copy()
                h_test[j] += delta * best_w[idx]
                h_test = np.maximum(h_test, 0)
                if np.sum(h_test) < 1e-15:
                    continue
                h_test = h_test / np.sum(h_test)
                test_score = _exact_ratio(h_test)
                if test_score > curr_score + 1e-10:
                    h_best = h_test.copy()
                    curr_score = test_score
                    improved = True
                    break
        
        if not improved:
            break
    
    return h_best, best_score

def _greedy_sidon_search(deadline):
    """
    Greedy search for a support set with low autocorrelation peak.
    """
    best_h = np.zeros(N)
    best_score = 0.0
    
    np.random.seed(42)
    
    max_k = 12 # Reduced max_k to allow more restarts within time
    num_starts = 20
    
    for start_id in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Random initial set of size 4-6
        init_size = 4 + (start_id % 3)
        current_indices = np.random.choice(N, size=init_size, replace=False)
        current_indices.sort()
        
        # Optimize weights on this initial set
        h_curr, curr_score = _optimize_weights(list(current_indices), deadline)
        
        if curr_score > best_score:
            best_score = curr_score
            best_h = h_curr.copy()
            
        # Greedy expansion
        for _ in range(max_k - len(current_indices)):
            if time.time() > deadline - 10:
                break
            
            best_new_index = -1
            best_new_score = curr_score
            best_new_h = h_curr.copy()
            
            candidates = [i for i in range(N) if i not in current_indices]
            
            # Quick metric: minimize max autocorrelation of the set (with unit weights)
            base_h = np.zeros(N)
            base_h[current_indices] = 1.0
            base_conv = fftconvolve(base_h, base_h, mode='full')
            
            # Find index j that minimizes the peak increase
            best_candidate_score = float('inf')
            best_candidate_idx = -1
            
            for j in candidates:
                h_test = base_h.copy()
                h_test[j] = 1.0
                conv_test = fftconvolve(h_test, h_test, mode='full')
                max_test = np.max(conv_test)
                if max_test < best_candidate_score:
                    best_candidate_score = max_test
                    best_candidate_idx = j
            
            if best_candidate_idx == -1:
                break
                
            new_indices = list(current_indices) + [best_candidate_idx]
            new_indices.sort()
            
            # Optimize weights on the new set
            h_optimized, new_score = _optimize_weights(new_indices, deadline)
            
            if new_score > best_new_score:
                curr_score = new_score
                current_indices = new_indices
                h_curr = h_optimized
                
            if curr_score > best_score:
                best_score = curr_score
                best_h = h_curr.copy()
                
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145 # Safety margin (150s limit)
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Phase 1: Evaluate pre-generated Sidon sets
    sidon_candidates = _generate_sidon_candidates()
    
    for support in sidon_candidates:
        if time.time() > deadline - 25:
            break
        h_opt, score = _optimize_weights(support, deadline)
        if score > best_score:
            best_score = score
            best_h = h_opt
    
    # Phase 2: Greedy search for potentially better supports
    if time.time() < deadline - 35:
        h_greedy, score_greedy = _greedy_sidon_search(deadline)
        if score_greedy > best_score:
            best_score = score_greedy
            best_h = h_greedy
            
    # Fallback if everything failed
    if best_score < 0.5:
        h_fallback = np.ones(N) / N
        score_fallback = _exact_ratio(h_fallback)
        if score_fallback > best_score:
            best_score = score_fallback
            best_h = h_fallback
            
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
