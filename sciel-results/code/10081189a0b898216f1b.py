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
    Uses analytical gradient for the smooth max surrogate.
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
            
            # 2. Gradient of smooth_max w.r.t w
            # smooth_max = (1/beta) log(sum_k exp(beta conv_k))
            # d smooth_max / d w_j = (1/beta) * (1/sum_exp) * sum_k exp(beta conv_k) * beta * d conv_k / d w_j
            # = sum_k (exp(beta conv_k) / sum_exp) * d conv_k / d w_j
            # Let p_k = exp(beta conv_k) / sum_exp (softmax weights on conv)
            # d conv_k / d w_j = 2 h_{k-j} (as derived in inspiration)
            # grad_smooth_max_j = 2 * sum_k p_k h_{k-j}
            
            exp_shifted = np.exp(beta * shifted)
            sum_exp = np.sum(exp_shifted)
            p = exp_shifted / sum_exp # Length 2N-1
            
            # Compute grad_smooth_max for each w_j (j in support)
            # grad_smooth_max[j] = 2 * sum_{k} p[k] * h[k-j]
            # Since h is zero outside support, h[k-j] is non-zero only if k-j in support.
            # Let m = k-j => k = m+j.
            # grad_smooth_max[j] = 2 * sum_{m in support} h[m] * p[m+j]
            
            grad_smooth = np.zeros(k)
            # For efficiency, precompute p
            for idx_j, j in enumerate(support):
                # Sum over m in support
                # h[m] is w_local[idx_m]
                # p[m+j]
                total = 0.0
                for idx_m, m in enumerate(support):
                    k_val = m + j
                    if 0 <= k_val < len(p):
                        total += w_local[idx_m] * p[k_val]
                grad_smooth[idx_j] = 2.0 * total
            
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
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
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
    # This helps snap to the true peak if the smooth max was slightly off
    h_best = np.zeros(N)
    h_best[support] = best_w
    
    for _ in range(50):
        if time.time() > deadline - 1:
            break
        improved = False
        for idx, j in enumerate(support):
            # Try adjusting w_j slightly
            # We can do a 1D line search or just try a few steps
            # Simpler: try setting w_j to 0 and see if it improves (sparsity increase)
            # Or just trust the optimizer. 
            # Let's do a simple local refinement by checking if moving mass helps.
            pass # Skipping complex coordinate descent to save time, optimizer is usually good enough with high beta.
            
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
    deadline = start_time + 140 # Safety margin (150s limit)
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Phase 1: Evaluate pre-generated Sidon sets
    sidon_candidates = _generate_sidon_candidates()
    
    for support in sidon_candidates:
        if time.time() > deadline - 20:
            break
        h_opt, score = _optimize_weights(support, deadline)
        if score > best_score:
            best_score = score
            best_h = h_opt
    
    # Phase 2: Greedy search for potentially better supports
    if time.time() < deadline - 30:
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
