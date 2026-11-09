# EVOLVE-BLOCK-START
import numpy as np
import time

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    Returns the probability as a float.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Build all pairwise sums s_i + s_j with probabilities p_i * p_j
    # We need: sum_{k,l} p_k p_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} p_i p_j)
    
    # Efficient approach using sorting and searchsorted:
    # 1. Compute all pair sums and their probabilities
    # 2. Sort by pair sum
    # 3. For each (k, l), threshold T = 2*s_l - s_k
    # 4. Use searchsorted to find CDF value at T
    # 5. P = sum_{k,l} p_k p_l * CDF(T)
    
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    pair_sums_flat = pair_sums.ravel()
    pair_probs_flat = pair_probs.ravel()
    
    # Sort by pair_sums
    sort_idx = np.argsort(pair_sums_flat)
    sorted_sums = pair_sums_flat[sort_idx]
    sorted_probs = pair_probs_flat[sort_idx]
    
    # Cumulative sum of probs
    cum_probs = np.cumsum(sorted_probs)
    
    # thresholds shape (n, n): thresholds[k, l] = 2*s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]
    T_flat = thresholds.ravel()
    
    # Find index where sorted_sums < T_flat (strict inequality)
    idx = np.searchsorted(sorted_sums, T_flat, side='left')
    
    cdf_vals = np.zeros(len(idx), dtype=np.float64)
    mask = idx > 0
    cdf_vals[mask] = cum_probs[idx[mask] - 1]
    
    cdf_vals = cdf_vals.reshape(n, n)
    
    # p_k p_l
    prob_kl = p[:, None] * p[None, :]
    
    P = np.sum(prob_kl * cdf_vals)
    
    return float(P)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = sup.copy()
            best_weights = wts.copy()
            print(f"New best: {score:.6f} at n={len(sup)}")
    
    def time_left():
        return deadline - time.time()
    
    # Strategy 1: Two-point distributions [0, 1] with weight p on 1
    # P = p * (1-p)^2 * (1+2p)
    # Maximize over p
    for p in np.linspace(0.0, 1.0, 2001):
        if time_left() < 10:
            break
        support = np.array([0.0, 1.0])
        weights = np.array([1.0 - p, p])
        score = _evaluate_measure(support, weights)
        update_best(support, weights, score)
    
    # Strategy 2: Three-point distributions [0, 1, x]
    # Try various x and weights
    np.random.seed(42)
    
    for x_val in np.linspace(0.5, 5.0, 20):
        if time_left() < 10:
            break
        # Random search over weights for fixed support [0, 1, x]
        support = np.array([0.0, 1.0, x_val])
        for _ in range(200):
            if time_left() < 10:
                break
            w = np.random.dirichlet(np.ones(3))
            score = _evaluate_measure(support, w)
            update_best(support, w, score)
    
    # Strategy 3: Geometric distributions on {0, 1, ..., n-1}
    for n in range(3, 20):
        if time_left() < 10:
            break
        for q in np.linspace(0.01, 0.99, 50):
            if time_left() < 10:
                break
            support = np.arange(n, dtype=np.float64)
            weights = np.power(q, support)
            if np.sum(weights) < 1e-9:
                continue
            score = _evaluate_measure(support, weights)
            update_best(support, weights, score)
    
    # Strategy 4: Power-law weights
    for n in range(3, 15):
        if time_left() < 10:
            break
        for alpha in np.linspace(0.1, 5.0, 50):
            if time_left() < 10:
                break
            support = np.arange(n, dtype=np.float64)
            weights = 1.0 / np.power(support + 1, alpha)
            score = _evaluate_measure(support, weights)
            update_best(support, weights, score)
    
    # Strategy 5: Random search with local optimization
    np.random.seed(123)
    
    # Try different n values
    for n in [3, 4, 5, 6, 7, 8, 10, 12]:
        if time_left() < 30:
            break
        
        # Multiple random starts with local hill climbing
        for start_iter in range(20):
            if time_left() < 30:
                break
            
            # Random support in [0, 2]
            support = np.sort(np.random.uniform(0, 2, n))
            # Random Dirichlet weights
            w = np.random.dirichlet(np.ones(n))
            
            # Local optimization: perturb weights
            best_local = _evaluate_measure(support, w)
            
            for it in range(100):
                if time_left() < 30:
                    break
                
                # Perturb weights
                w_new = w + np.random.randn(n) * 0.05
                w_new = np.maximum(w_new, 0)
                w_sum = np.sum(w_new)
                if w_sum < 1e-9:
                    continue
                w_new = w_new / w_sum
                
                score_new = _evaluate_measure(support, w_new)
                
                if score_new > best_local:
                    w = w_new
                    best_local = score_new
            
            update_best(support, w, best_local)
    
    # Strategy 6: Optimize support for fixed n with random weights
    for n in [3, 4, 5]:
        if time_left() < 30:
            break
        
        for _ in range(10):
            if time_left() < 30:
                break
            
            # Random weights
            w = np.random.dirichlet(np.ones(n))
            # Random support
            support = np.sort(np.random.uniform(0, 2, n))
            
            best_local = _evaluate_measure(support, w)
            
            for it in range(50):
                if time_left() < 30:
                    break
                
                # Perturb support
                s_new = support + np.random.randn(n) * 0.1
                s_new = np.maximum(s_new, 0)
                s_new = np.sort(s_new)
                
                score_new = _evaluate_measure(s_new, w)
                
                if score_new > best_local:
                    support = s_new
                    best_local = score_new
            
            update_best(support, w, best_local)
    
    # Strategy 7: Finer search around best found so far
    if best_support is not None:
        n_best = len(best_support)
        if time_left() > 60:
            # Refine around the best solution
            for _ in range(50):
                if time_left() < 30:
                    break
                
                # Perturb both support and weights
                s_new = best_support + np.random.randn(n_best) * 0.05
                s_new = np.maximum(s_new, 0)
                s_new = np.sort(s_new)
                
                w_new = best_weights + np.random.randn(n_best) * 0.05
                w_new = np.maximum(w_new, 0)
                w_sum = np.sum(w_new)
                if w_sum < 1e-9:
                    continue
                w_new = w_new / w_sum
                
                score = _evaluate_measure(s_new, w_new)
                update_best(s_new, w_new, score)
    
    # Fallback if nothing found
    if best_support is None:
        support = np.array([0.0, 1.0])
        weights = np.array([0.5, 0.5])
        best_support = support
        best_weights = weights
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
