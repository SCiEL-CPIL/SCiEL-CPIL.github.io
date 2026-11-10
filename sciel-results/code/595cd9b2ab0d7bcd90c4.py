import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.optimize import differential_evolution

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for a discrete measure.
    support: 1D array of shape (n,)
    weights: 1D array of shape (n,)
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    
    # Normalize weights just in case, though optimization should keep them normalized
    total_w = w.sum()
    if total_w <= 1e-12:
        return 0.0
    w = w / total_w
    
    # 1. Compute all pairwise sums for X1 + X2
    # s_i + s_j
    ss = s[:, None] + s[None, :]
    ww = w[:, None] * w[None, :]
    
    # Flatten and sort
    pair_sums = ss.flatten()
    pair_weights = ww.flatten()
    
    order = np.argsort(pair_sums, kind='stable')
    sorted_sums = pair_sums[order]
    sorted_weights = pair_weights[order]
    cum_weights = np.cumsum(sorted_weights)
    
    # 2. Compute P = sum_{k,l} w_k w_l F(2*s_l - s_k)
    # Thresholds matrix T[k, l] = 2*s_l - s_k
    # Note: s is (n,).
    # T shape (n, n) where T[k, l] corresponds to X3=s_k, X4=s_l
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # For each threshold, find cumulative weight of pair_sums < threshold
    # searchsorted returns index of first element >= x.
    # We want < x, so index is count of elements < x.
    idxs = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Get cumulative weights
    # If idx > 0, cum_weight is cum_weights[idx-1]
    # If idx == 0, cum_weight is 0
    idxs_clipped = np.clip(idxs - 1, 0, len(cum_weights) - 1)
    probs = np.where(idxs > 0, cum_weights[idxs_clipped], 0.0)
    
    # P = sum_{k,l} w_k w_l probs[k,l]
    # w[:, None] is (n, 1), w[None, :] is (1, n). Product is (n, n)
    P = np.sum(w[:, None] * w[None, :] * probs)
    
    return float(P)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600.0
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Helper to update best
    def update_best(s, w, score):
        nonlocal best_support, best_weights, best_score
        if score > best_score:
            best_support = s.copy()
            best_weights = w.copy()
            best_score = score
            # print(f"New best: {score:.6f} at n={len(s)}")

    # Try different n values with differential evolution
    for n in [6, 8, 10]:
        if time.time() > deadline - 30.0:
            break
        
        # Parameters: support (n values in [0, 20]) and log-weights (n values in [-10, 10])
        # Total 2n parameters
        
        def objective(params):
            s_opt = np.asarray(params[:n], dtype=np.float64)
            log_w_opt = np.asarray(params[n:], dtype=np.float64)
            
            # Support must be non-negative; params are already bounded [0, 20]
            s_pos = s_opt
            
            # Weights via softmax
            w_opt = softmax(log_w_opt)
            
            P = _compute_P(s_pos, w_opt)
            return -P
        
        # Bounds for differential evolution
        bounds = [(0.0, 20.0)] * n + [(-10.0, 10.0)] * n
        
        # Run differential evolution with time-limited iterations
        max_time = min(150.0, deadline - time.time() - 30.0)
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.7,
                seed=42,
                polish=True,
                options={'maxiter': 200}
            )
            
            if result.fun < 0 or result.fun < -1e-10:
                s_opt = result.x[:n]
                log_w_opt = result.x[n:]
                
                s_final = s_opt
                w_final = softmax(log_w_opt)
                
                # Ensure validity
                if np.any(s_final < -1e-9) or np.any(w_final < -1e-9):
                    continue
                if w_final.sum() <= 1e-9:
                    continue
                    
                score = _compute_P(s_final, w_final)
                update_best(s_final, w_final, score)
                
        except Exception:
            pass

    # Also try a quick L-BFGS-B refinement on the best solution found
    if best_support is not None and time.time() < deadline - 60.0:
        n = len(best_support)
        
        def objective_refine(params):
            s_opt = np.asarray(params[:n], dtype=np.float64)
            log_w_opt = np.asarray(params[n:], dtype=np.float64)
            
            s_pos = np.abs(s_opt)  # Ensure non-negative
            w_opt = softmax(log_w_opt)
            
            P = _compute_P(s_pos, w_opt)
            return -P
        
        s_params_init = best_support.copy()
        w_params_init = np.log(best_weights + 1e-10)
        x0 = np.concatenate([s_params_init, w_params_init])
        
        try:
            result = minimize(objective_refine, x0, method='L-BFGS-B',
                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if result.fun < objective_refine(x0):
                s_opt = np.abs(result.x[:n])
                log_w_opt = result.x[n:]
                
                s_final = s_opt
                w_final = softmax(log_w_opt)
                
                if np.any(s_final < -1e-9) or np.any(w_final < -1e-9):
                    pass
                elif w_final.sum() <= 1e-9:
                    pass
                else:
                    score = _compute_P(s_final, w_final)
                    update_best(s_final, w_final, score)
        except Exception:
            pass

    # Fallback if no valid solution found
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    return best_support.astype(np.float64), best_weights.astype(np.float64)
