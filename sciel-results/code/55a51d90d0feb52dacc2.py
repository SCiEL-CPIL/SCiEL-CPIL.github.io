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
    Uses direct parameterization with non-negativity constraints.
    """
    k = len(support)
    if k == 0:
        return np.zeros(N), 0.0
    
    w_init = np.ones(k) / k
    h_init = np.zeros(N)
    h_init[support] = w_init
    best_score = _exact_ratio(h_init)
    best_w = w_init.copy()
    
    # Smooth max with moderate beta for optimization stability
    beta_opt = 100.0
    
    def _neg_ratio_smooth(params):
        """Minimize -ratio with h_i >= 0 constraint handled by bounds."""
        w_local = params / np.sum(params)  # Normalize
        h_local = np.zeros(N)
        h_local[support] = w_local
        
        l2_sq = np.sum(h_local**2)
        conv = fftconvolve(h_local, h_local, mode='full')
        
        m = np.max(conv)
        if m < 1e-15:
            return -1e-10, np.zeros(k)
        shifted = conv - m
        shifted = np.clip(shifted, -500, 500)
        log_sum_exp = m + np.log(np.sum(np.exp(beta_opt * shifted)))
        smooth_max = log_sum_exp / beta_opt
        
        if smooth_max < 1e-15:
            return -1e-10, np.zeros(k)
            
        obj_val = -l2_sq / smooth_max
        
        # Gradient computation
        exp_shifted = np.exp(beta_opt * shifted)
        sum_exp = np.sum(exp_shifted)
        p = exp_shifted / sum_exp
        
        # d smooth_max / d h_j = 2 * sum_t p[t+j] * h[t]
        # Compute via FFT cross-correlation
        fft_size = 1
        while fft_size < 3 * N:
            fft_size *= 2
            
        p_padded = np.zeros(fft_size)
        p_padded[:2*N-1] = p
        h_padded = np.zeros(fft_size)
        h_padded[:N] = h_local
        
        P_fft = np.fft.rfft(p_padded)
        H_fft = np.fft.rfft(h_padded)
        cross_corr = np.fft.irfft(P_fft * np.conj(H_fft))
        
        grad_smooth = np.zeros(N)
        for j in support:
            grad_smooth[j] = 2.0 * cross_corr[j]
        
        grad_l2 = 2.0 * h_local
        grad_l2_full = np.zeros(N)
        for j, idx in enumerate(support):
            grad_l2_full[j] = grad_l2[idx]
        
        grad_w = (l2_sq * grad_smooth - smooth_max * grad_l2_full) / (smooth_max ** 2)
        
        # Chain rule for normalization: d/dw_j (f(g(w)/sum(g)))
        # Simplified: treat params directly
        grad_params = np.zeros(k)
        for i in range(k):
            grad_params[i] = grad_w[support[i]]
        # Adjust for normalization
        avg_grad = np.mean(grad_params)
        grad_params = grad_params - avg_grad
        
        return obj_val, grad_params

    # Use multiple initializations and betas
    for beta_try in [50.0, 100.0, 200.0, 300.0]:
        if time.time() > deadline - 3:
            break
        beta_opt = beta_try
        
        # Multiple random starts
        for start in range(5):
            if time.time() > deadline - 3:
                break
            if start == 0:
                w_curr = w_init.copy()
            else:
                np.random.seed(start * 7 + len(support))
                w_curr = np.random.dirichlet(np.ones(k))
            
            bounds = [(1e-10, 1.0)] * k
            
            try:
                res = minimize(
                    _neg_ratio_smooth,
                    w_curr.copy(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                w_res = res.x / np.sum(res.x)
                h_res = np.zeros(N)
                h_res[support] = w_res
                score = _exact_ratio(h_res)
                if score > best_score:
                    best_score = score
                    best_w = w_res.copy()
                    w_init = w_res.copy()
            except Exception:
                pass
    
    # Final local polish with exact objective
    h_best = np.zeros(N)
    h_best[support] = best_w
    curr_score = best_score
    
    for _ in range(50):
        if time.time() > deadline - 1:
            break
        improved = False
        for idx, j in enumerate(support):
            for scale in [0.5, 0.8, 1.2, 1.5, 2.0]:
                h_test = h_best.copy()
                h_test[j] *= scale
                h_test = np.maximum(h_test, 0)
                total = np.sum(h_test)
                if total < 1e-15:
                    continue
                h_test = h_test / total
                test_score = _exact_ratio(h_test)
                if test_score > curr_score + 1e-12:
                    h_best = h_test.copy()
                    best_w = h_best[support].copy()
                    curr_score = test_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return h_best, curr_score

def _local_search_polish(h, deadline, max_iters=200):
    """
    Coordinate descent and pair-swap local search on the exact objective.
    """
    h = np.maximum(h, 0).copy()
    total = np.sum(h)
    if total < 1e-15:
        return h
    h = h / total
    curr_score = _exact_ratio(h)
    
    for _ in range(max_iters):
        if time.time() > deadline - 1:
            break
        improved = False
        
        # Coordinate scaling
        for i in range(N):
            if h[i] < 1e-12:
                continue
            for scale in [0.7, 0.85, 1.15, 1.3]:
                h_test = h.copy()
                h_test[i] *= scale
                h_test = np.maximum(h_test, 0)
                t = np.sum(h_test)
                if t < 1e-15:
                    continue
                h_test = h_test / t
                s = _exact_ratio(h_test)
                if s > curr_score + 1e-12:
                    h = h_test.copy()
                    curr_score = s
                    improved = True
                    break
        
        # Pair swaps: transfer mass from i to j
        support = np.where(h > 1e-6)[0]
        for i in range(len(support)):
            for j in range(i+1, len(support)):
                a, b = support[i], support[j]
                for frac in [0.1, 0.3, 0.5]:
                    delta = h[a] * frac
                    h_test = h.copy()
                    h_test[a] -= delta
                    h_test[b] += delta
                    h_test = np.maximum(h_test, 0)
                    t = np.sum(h_test)
                    if t < 1e-15:
                        continue
                    h_test = h_test / t
                    s = _exact_ratio(h_test)
                    if s > curr_score + 1e-12:
                        h = h_test.copy()
                        curr_score = s
                        improved = True
                        break
        
        if not improved:
            break
    
    return h

def _greedy_sidon_search(deadline):
    """
    Greedy search for a support set with low autocorrelation peak.
    """
    best_h = np.zeros(N)
    best_score = 0.0
    
    np.random.seed(42)
    
    max_k = 12
    num_starts = 15
    
    for start_id in range(num_starts):
        if time.time() > deadline - 15:
            break
        
        init_size = 4 + (start_id % 3)
        current_indices = np.random.choice(N, size=init_size, replace=False)
        current_indices.sort()
        
        h_curr, curr_score = _optimize_weights(list(current_indices), deadline)
        
        if curr_score > best_score:
            best_score = curr_score
            best_h = h_curr.copy()
        
        for _ in range(max_k - len(current_indices)):
            if time.time() > deadline - 15:
                break
            
            base_h = np.zeros(N)
            base_h[current_indices] = 1.0
            base_conv = fftconvolve(base_h, base_h, mode='full')
            
            best_candidate_score = float('inf')
            best_candidate_idx = -1
            candidates = [i for i in range(N) if i not in current_indices]
            
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
            
            h_optimized, new_score = _optimize_weights(new_indices, deadline)
            
            if new_score > curr_score:
                curr_score = new_score
                current_indices = new_indices
                h_curr = h_optimized
            
            if curr_score > best_score:
                best_score = curr_score
                best_h = h_curr.copy()
                
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 143
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Phase 1: Evaluate pre-generated Sidon sets
    sidon_candidates = _generate_sidon_candidates()
    
    for support in sidon_candidates:
        if time.time() > deadline - 40:
            break
        h_opt, score = _optimize_weights(support, deadline)
        if score > best_score:
            best_score = score
            best_h = h_opt
    
    # Phase 2: Greedy search
    if time.time() < deadline - 45:
        h_greedy, score_greedy = _greedy_sidon_search(deadline)
        if score_greedy > best_score:
            best_score = score_greedy
            best_h = h_greedy
    
    # Phase 3: Local search polish on best result
    if time.time() < deadline - 5:
        best_h = _local_search_polish(best_h, deadline)
        best_score = _exact_ratio(best_h)
    
    # Fallback
    if best_score < 0.5:
        h_fallback = np.ones(N) / N
        score_fallback = _exact_ratio(h_fallback)
        if score_fallback > best_score:
            best_score = score_fallback
            best_h = h_fallback
    
    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
