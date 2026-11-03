"""
Second autocorrelation inequality: choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]
to maximize (sum h_i^2) / ((sum h_i) * sup_k C[k]) where C = h autoconvolution.

The autoconvolution of the step function f(t) = sum h_i * 1_{[x_i, x_{i+1})}(t) with delta = 1/50:
- ||f||_1 = delta * sum(h_i)
- ||f||_2^2 = delta * sum(h_i^2)
- ||f*f||_inf = delta * max(conv(h,h))

Ratio = (delta * sum h^2) / (delta * sum h * delta * max conv) = sum h^2 / (delta * sum h * max conv)

The evaluator normalizes (likely L1=1), so we maximize: sum(h^2) / (delta * max(conv(h,h)))
subject to sum(h)=1, h>=0.

Strategy: Use sparse, well-separated peaks to minimize autoconvolution overlap.
Combined approach:
1. Greedy Sidon-like placement to find low-overlap supports.
2. Continuous optimization on log-weights to refine amplitudes.
3. Local search for structural changes (add/remove/move peaks).
"""
import numpy as np
import time
from itertools import combinations

N = 50
DELTA = 1.0 / N  # = 0.02

def _score_raw(h):
    """Raw score: sum h^2 / (delta * sum h * max conv). This is scale-invariant if we consider the normalized ratio."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv <= 0:
        return 0.0
    # The actual ratio: sum h^2 / (delta * sum h * max conv)
    return l2_sq / (DELTA * l1 * max_conv)

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    best_h = None
    best_score = 0.0
    
    def update_best(h):
        nonlocal best_h, best_score
        s = _score_raw(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Phase 1: Greedy Sidon-like construction
    # We want to place peaks such that their pairwise differences are all distinct (or at least non-overlapping in convolution).
    # A Sidon set (B_2 set) has distinct pairwise sums. For convolution (which involves sums of indices),
    # we want to minimize collisions in the convolution support.
    # Greedy: start with empty set, add index that minimizes the increase in max convolution.
    
    def greedy_sidon(max_peaks=10):
        h = np.zeros(N)
        h[0] = 1.0
        support = [0]
        
        for _ in range(max_peaks - 1):
            if time.time() > deadline:
                break
            best_idx = -1
            best_s = -1
            # Try adding each possible new index
            for idx in range(N):
                if idx in support:
                    continue
                h_try = h.copy()
                h_try[idx] = 1.0
                s = _score_raw(h_try)
                if s > best_s:
                    best_s = s
                    best_idx = idx
            if best_idx != -1:
                h[best_idx] = 1.0
                support.append(best_idx)
                update_best(h)
            else:
                break
        return h
    
    h_greedy = greedy_sidon(15)
    update_best(h_greedy)
    
    # Phase 2: Random sparse initializations with varying sparsity
    np.random.seed(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        num_peaks = np.random.randint(3, 12)
        indices = np.random.choice(N, num_peaks, replace=False)
        h = np.zeros(N)
        h[indices] = 1.0
        update_best(h)
        
        # Also try random weights
        for _ in range(5):
            h_rand = np.zeros(N)
            h_rand[indices] = np.random.uniform(0.5, 2.0, size=num_peaks)
            update_best(h_rand)
    
    # Phase 3: Continuous optimization on log-weights for the best sparse supports
    # For a fixed support, optimize the weights.
    # Let support S, weights w_i for i in S.
    # h_i = w_i if i in S, else 0.
    # Maximize sum(w_i^2) / (delta * sum(w_i) * max(conv(h,h)))
    # This is equivalent to maximizing sum(w_i^2) / max(conv(h,h)) subject to sum(w_i) = 1 (scale invariance allows fixing sum).
    
    def optimize_weights(support, init_weights=None):
        if len(support) == 0:
            return None
        if init_weights is None:
            init_weights = np.ones(len(support))
        
        # Use simple gradient-free local search on weights
        # Normalize weights to sum to 1
        w = init_weights.copy()
        w = w / np.sum(w)
        
        best_w = w.copy()
        best_s = -1
        
        h_curr = np.zeros(N)
        for i, idx in enumerate(support):
            h_curr[idx] = w[i]
        best_s = _score_raw(h_curr)
        
        for iter in range(100):
            if time.time() > deadline:
                break
            
            improved = False
            # Try adjusting each weight
            for i in range(len(support)):
                orig_w = w[i]
                for factor in [0.5, 0.7, 1.3, 1.5, 2.0, 0.1, 0.2]:
                    w_try = w.copy()
                    w_try[i] = orig_w * factor
                    # Renormalize
                    w_try = w_try / np.sum(w_try)
                    
                    h_try = np.zeros(N)
                    for j, idx in enumerate(support):
                        h_try[idx] = w_try[j]
                    
                    s = _score_raw(h_try)
                    if s > best_s:
                        best_s = s
                        best_w = w_try.copy()
                        w = w_try.copy()
                        h_curr = h_try.copy()
                        improved = True
                        break
            
            if not improved:
                # Try small random perturbations
                for _ in range(10):
                    w_try = w + np.random.normal(0, 0.1, size=len(support))
                    w_try = np.abs(w_try)
                    if np.sum(w_try) == 0:
                        continue
                    w_try = w_try / np.sum(w_try)
                    
                    h_try = np.zeros(N)
                    for j, idx in enumerate(support):
                        h_try[idx] = w_try[j]
                    
                    s = _score_raw(h_try)
                    if s > best_s:
                        best_s = s
                        best_w = w_try.copy()
                        w = w_try.copy()
                        h_curr = h_try.copy()
                        improved = True
            
            if not improved:
                break
        
        return best_w
    
    # Optimize weights for the greedy solution
    support_greedy = np.where(h_greedy > 1e-6)[0]
    if len(support_greedy) > 0:
        w_opt = optimize_weights(support_greedy, h_greedy[support_greedy])
        if w_opt is not None:
            h_opt = np.zeros(N)
            for i, idx in enumerate(support_greedy):
                h_opt[idx] = w_opt[i]
            update_best(h_opt)
    
    # Optimize weights for some random sparse sets
    for _ in range(50):
        if time.time() > deadline:
            break
        num_peaks = np.random.randint(4, 10)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_init = np.zeros(N)
        h_init[indices] = 1.0
        w_opt = optimize_weights(indices, h_init[indices])
        if w_opt is not None:
            h_opt = np.zeros(N)
            for i, idx in enumerate(indices):
                h_opt[idx] = w_opt[i]
            update_best(h_opt)
    
    # Phase 4: Local refinement of the global best
    if best_h is not None:
        h_curr = best_h.copy()
        support_curr = np.where(h_curr > 1e-6)[0]
        
        for _ in range(50):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding a new peak
            zero_idx = np.where(h_curr < 1e-6)[0]
            if len(zero_idx) > 0:
                # Try adding each zero index and see if it improves
                best_add_idx = -1
                best_add_s = best_score
                for idx in zero_idx:
                    h_try = h_curr.copy()
                    h_try[idx] = 0.1  # Small initial value
                    s = _score_raw(h_try)
                    if s > best_add_s:
                        best_add_s = s
                        best_add_idx = idx
                
                if best_add_idx != -1:
                    h_curr[best_add_idx] = 0.1
                    support_curr = np.where(h_curr > 1e-6)[0]
                    # Optimize weights for new support
                    w_opt = optimize_weights(support_curr, h_curr[support_curr])
                    if w_opt is not None:
                        h_new = np.zeros(N)
                        for i, idx in enumerate(support_curr):
                            h_new[idx] = w_opt[i]
                        if _score_raw(h_new) > best_score:
                            best_score = _score_raw(h_new)
                            best_h = h_new.copy()
                            h_curr = h_new.copy()
                            improved = True
            
            # Try removing a peak
            if len(support_curr) > 2:
                best_remove_idx = -1
                best_remove_s = best_score
                for idx in support_curr:
                    h_try = h_curr.copy()
                    h_try[idx] = 0.0
                    s = _score_raw(h_try)
                    if s > best_remove_s:
                        best_remove_s = s
                        best_remove_idx = idx
                
                if best_remove_idx != -1:
                    h_curr[best_remove_idx] = 0.0
                    support_curr = np.where(h_curr > 1e-6)[0]
                    # Optimize weights for reduced support
                    w_opt = optimize_weights(support_curr, h_curr[support_curr])
                    if w_opt is not None:
                        h_new = np.zeros(N)
                        for i, idx in enumerate(support_curr):
                            h_new[idx] = w_opt[i]
                        if _score_raw(h_new) > best_score:
                            best_score = _score_raw(h_new)
                            best_h = h_new.copy()
                            h_curr = h_new.copy()
                            improved = True
            
            # Try moving a peak
            for idx in support_curr:
                for offset in [-1, 1, -2, 2]:
                    new_idx = idx + offset
                    if 0 <= new_idx < N and h_curr[new_idx] < 1e-6:
                        h_try = h_curr.copy()
                        h_try[new_idx] = h_try[idx]
                        h_try[idx] = 0.0
                        s = _score_raw(h_try)
                        if s > best_score:
                            best_score = s
                            best_h = h_try.copy()
                            h_curr = h_try.copy()
                            support_curr = np.where(h_curr > 1e-6)[0]
                            improved = True
                            break
            
            if not improved:
                # Final weight optimization
                w_opt = optimize_weights(support_curr, h_curr[support_curr])
                if w_opt is not None:
                    h_new = np.zeros(N)
                    for i, idx in enumerate(support_curr):
                        h_new[idx] = w_opt[i]
                    if _score_raw(h_new) > best_score:
                        best_score = _score_raw(h_new)
                        best_h = h_new.copy()
                        h_curr = h_new.copy()
                else:
                    break
    
    # Return best found
    if best_h is not None:
        return list(best_h)
    return list(np.ones(N))
