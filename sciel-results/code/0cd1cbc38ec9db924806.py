# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_prob(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for given support and normalized weights."""
    n = len(support)
    w = weights / weights.sum()
    s = support
    
    # Precompute pairwise sums: s[i] + s[j] for all i,j
    # For each (k, l), we need sum over (i,j) with s[i]+s[j] < 2*s[l] - s[k]
    # P = sum_{k,l} w[k]*w[l] * (sum_{i,j: s[i]+s[j] < 2*s[l]-s[k]} w[i]*w[j])
    
    # Efficient approach: for each l, compute threshold t_l = 2*s[l]
    # For each k, threshold is t_l - s[k]
    # We need cumulative sum of w[i]*w[j] over s[i]+s[j] < threshold
    
    # Precompute all pairwise sums and their probability masses
    # s[i] + s[j] for i,j in 0..n-1
    ss = s[:, None] + s[None, :]  # n x n
    ww = w[:, None] * w[None, :]  # n x n
    
    # Sort by sum value
    flat_ss = ss.ravel()
    flat_ww = ww.ravel()
    
    # For efficient lookup, we can use searchsorted on sorted sums
    sort_idx = np.argsort(flat_ss, kind='mergesort')
    sorted_ss = flat_ss[sort_idx]
    sorted_ww = flat_ww[sort_idx]
    cum_ww = np.cumsum(sorted_ww)
    
    P = 0.0
    for l in range(n):
        if w[l] == 0:
            continue
        for k in range(n):
            if w[k] == 0:
                continue
            threshold = 2.0 * s[l] - s[k]
            # Find count of pairs with sum < threshold (strict)
            idx = np.searchsorted(sorted_ss, threshold, side='left')
            if idx > 0:
                pair_prob = cum_ww[idx - 1]
            else:
                pair_prob = 0.0
            P += w[k] * w[l] * pair_prob
    
    return P


def _compute_prob_fast(support, weights):
    """Faster computation using vectorized operations."""
    n = len(support)
    w = weights / weights.sum()
    s = support
    
    # For each pair (i,j), compute s[i]+s[j]
    ss = s[:, None] + s[None, :]
    ww = w[:, None] * w[None, :]
    
    # For each (k,l), threshold = 2*s[l] - s[k]
    # Create threshold matrix: thresholds[k,l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]  # n x n, [k,l]
    
    # For each (k,l), we need sum of ww[i,j] where ss[i,j] < thresholds[k,l]
    # This is a 4D problem: sum over i,j,k,l of w[k]*w[l]*w[i]*w[j]*[ss[i,j] < 2*s[l]-s[k]]
    
    # Vectorize: for each (i,j) pair, find which (k,l) satisfy the condition
    # ss[i,j] < 2*s[l] - s[k]  =>  s[k] + ss[i,j] < 2*s[l]  =>  s[k] + s[i] + s[j] < 2*s[l]
    
    # Let u[i,j,k] = s[i] + s[j] + s[k], and we need u[i,j,k] < 2*s[l]
    # P = sum_{i,j,k,l} w[i]*w[j]*w[k]*w[l] * [s[i]+s[j]+s[k] < 2*s[l]]
    
    # Compute all triple sums
    triple_sums = s[:, None, None] + s[None, :, None] + s[None, None, :]  # n^3
    triple_weights = w[:, None, None] * w[None, :, None] * w[None, None, :]  # n^3
    
    # For each l, count triples with sum < 2*s[l]
    two_s_l = 2.0 * s  # n
    # triple_sums shape (n,n,n), two_s_l shape (n,)
    # For each l, mask = triple_sums < two_s_l[l]
    # P = sum_l w[l] * sum_{i,j,k: triple_sums[i,j,k] < 2*s[l]} triple_weights[i,j,k]
    
    P = 0.0
    for l in range(n):
        if w[l] == 0:
            continue
        mask = triple_sums < two_s_l[l]
        P += w[l] * np.sum(triple_weights[mask])
    
    return P


def _neg_prob(params, n_points):
    """Objective function for optimization. params = [s_0, s_1, ..., s_{n-1}, w_0, w_1, ..., w_{n-1}]"""
    s = params[:n_points]
    w = params[n_points:]
    
    # Enforce non-negativity
    if np.any(s < -1e-10) or np.any(w < -1e-10):
        return 1e10
    
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    
    if np.sum(w) < 1e-10:
        return 1e10
    
    try:
        p = _compute_prob_fast(s, w)
        return -p
    except:
        return 1e10


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin
    
    best_support = None
    best_weights = None
    best_score = 0.0
    
    # Start with the 2-point optimal solution
    p_opt = (15 - np.sqrt(33)) / 16.0
    s2 = np.array([0.0, 1.0])
    w2 = np.array([p_opt, 1.0 - p_opt])
    
    try:
        score2 = _compute_prob_fast(s2, w2)
        best_score = score2
        best_support = s2.copy()
        best_weights = w2.copy()
    except:
        pass
    
    # Try 3-point supports with optimization
    for n_pts in [3, 4, 5]:
        if time.time() > deadline:
            break
        
        # Multiple random starts
        n_starts = 20 if n_pts <= 4 else 10
        for start_i in range(n_starts):
            if time.time() > deadline:
                break
            
            if start_i == 0:
                # Start from 2-point solution extended
                if n_pts == 3:
                    s_init = np.array([0.0, 0.5, 1.0])
                    w_init = np.array([p_opt * 0.5, 1.0 - p_opt, p_opt * 0.5])
                elif n_pts == 4:
                    s_init = np.array([0.0, 0.33, 0.67, 1.0])
                    w_init = np.array([0.3, 0.2, 0.2, 0.3])
                else:
                    s_init = np.linspace(0, 1, n_pts)
                    w_init = np.ones(n_pts) / n_pts
            else:
                # Random initialization
                s_init = np.sort(np.random.uniform(0, 2, n_pts))
                s_init[0] = 0.0  # Always include 0
                w_init = np.random.dirichlet(np.ones(n_pts))
            
            params_init = np.concatenate([s_init, w_init])
            
            try:
                result = minimize(
                    _neg_prob,
                    params_init,
                    args=(n_pts,),
                    method='L-BFGS-B',
                    bounds=[(0, 10)] * n_pts + [(0, 1)] * n_pts,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                
                if result.success or result.fun < -best_score:
                    s_opt = np.maximum(result.x[:n_pts], 0.0)
                    w_opt = np.maximum(result.x[n_pts:], 0.0)
                    w_opt = w_opt / w_opt.sum()
                    
                    score = _compute_prob_fast(s_opt, w_opt)
                    if score > best_score:
                        best_score = score
                        best_support = s_opt.copy()
                        best_weights = w_opt.copy()
            except:
                pass
    
    # Local refinement: try small perturbations of the best solution
    if best_support is not None and time.time() < deadline:
        n_pts = len(best_support)
        for trial in range(50):
            if time.time() > deadline:
                break
            
            # Random perturbation
            s_pert = best_support + np.random.normal(0, 0.01, n_pts)
            w_pert = best_weights + np.random.normal(0, 0.01, n_pts)
            s_pert = np.maximum(s_pert, 0.0)
            w_pert = np.maximum(w_pert, 0.0)
            
            if w_pert.sum() < 1e-10:
                continue
            
            try:
                score = _compute_prob_fast(s_pert, w_pert)
                if score > best_score:
                    best_score = score
                    best_support = s_pert.copy()
                    best_weights = w_pert / w_pert.sum()
            except:
                pass
    
    # Ensure validity
    if best_support is None or best_weights is None:
        # Fallback to uniform
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    # Clean up: remove points with negligible weight
    n_pts = len(best_support)
    if n_pts > 1:
        min_weight = 1e-10
        keep = best_weights > min_weight
        if np.sum(keep) >= 1:
            best_support = best_support[keep]
            best_weights = best_weights[keep]
    
    # Final validation
    if np.any(np.isnan(best_support)) or np.any(np.isinf(best_support)):
        best_support = np.arange(32, dtype=np.float64)
        best_weights = np.ones(32, dtype=np.float64) / 32
    
    if np.any(np.isnan(best_weights)) or np.any(np.isinf(best_weights)):
        best_support = np.arange(32, dtype=np.float64)
        best_weights = np.ones(32, dtype=np.float64) / 32
    
    if np.any(best_support < -1e-9):
        best_support = np.maximum(best_support, 0.0)
    
    if np.any(best_weights < -1e-9):
        best_weights = np.maximum(best_weights, 0.0)
    
    if best_weights.sum() <= 1e-9:
        best_support = np.arange(32, dtype=np.float64)
        best_weights = np.ones(32, dtype=np.float64) / 32
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
