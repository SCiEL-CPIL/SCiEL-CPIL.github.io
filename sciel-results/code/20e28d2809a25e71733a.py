# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for i.i.d. X from the discrete measure.
    P = sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i w_j)
    """
    n = len(support)
    # Normalize weights
    w = weights / np.sum(weights)
    
    # Build pairwise sums matrix: pair_sums[i, j] = s_i + s_j
    pair_sums = support[:, None] + support[None, :]
    
    # Threshold matrix: thresh[k, l] = 2*s_l - s_k
    # For each (k, l), we need sum of w_i*w_j where pair_sums[i,j] < thresh[k,l]
    # thresh has shape (n, n) where first index is k, second is l
    thresh = 2.0 * support[None, :] - support[:, None]
    # thresh[k, l] = 2*s_l - s_k
    
    # For each pair (k, l), count weighted pairs (i, j) with s_i + s_j < 2*s_l - s_k
    # This is a 4D problem but we can vectorize over (k, l)
    # inner_prob[k, l] = sum_{i,j} w_i w_j * 1[s_i+s_j < 2*s_l - s_k]
    
    # pair_sums shape (n, n), thresh shape (n, n)
    # We need for each (k, l): sum over (i, j) of w_i*w_j where pair_sums[i,j] < thresh[k,l]
    
    # Let's use a sorted approach for efficiency
    # Flatten pair_sums and sort
    flat_pairs = pair_sums.ravel()
    flat_weights = (w[:, None] * w[None, :]).ravel()
    
    # Sort by pair sum
    sort_idx = np.argsort(flat_pairs, kind='mergesort')
    sorted_pairs = flat_pairs[sort_idx]
    sorted_weights = flat_weights[sort_idx]
    cumsum_weights = np.cumsum(sorted_weights)
    
    # For each (k, l), find how many pairs have sum < thresh[k,l]
    # Use searchsorted
    thresh_flat = thresh.ravel()  # shape (n*n,)
    # We want count of pairs with sum < thresh
    # searchsorted returns the index where thresh would be inserted to maintain order
    # Since we want strict inequality (<), we use side='left'
    indices = np.searchsorted(sorted_pairs, thresh_flat, side='left')
    
    # For each index, get the cumulative sum of weights
    # If index == 0, sum is 0; otherwise cumsum_weights[index-1]
    inner_probs = np.where(indices > 0, cumsum_weights[np.maximum(indices - 1, 0)], 0.0)
    
    # Reshape inner_probs to (n, n) where first index is k, second is l
    inner_probs = inner_probs.reshape(n, n)
    
    # Now P = sum_{k,l} w_k * w_l * inner_probs[k, l]
    # w_k w_l is outer product
    P = np.sum(w[:, None] * w[None, :] * inner_probs)
    
    return P

def _neg_P(params, n):
    """
    Objective function for optimization.
    params: first n elements are support, next n are weights.
    """
    support = params[:n]
    weights = params[n:]
    
    # Ensure support is non-negative
    support = np.maximum(support, 0.0)
    # Ensure weights are non-negative
    weights = np.maximum(weights, 1e-10)
    
    return -_compute_P(support, weights)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    
    # Start with baseline
    n_baseline = 32
    support_baseline = np.arange(n_baseline, dtype=np.float64)
    weights_baseline = np.ones(n_baseline, dtype=np.float64) / n_baseline
    best_P = _compute_P(support_baseline, weights_baseline)
    best_support = support_baseline.copy()
    best_weights = weights_baseline.copy()
    
    # Try different sizes and optimization approaches
    # We'll use a combination of differential evolution and local refinement
    
    # Strategy: Try several values of n, and for each, run optimization
    n_values = [16, 32, 48, 64]
    
    for n in n_values:
        if time.time() > deadline - 30:  # Leave 30s buffer
            break
        
        # Initial guess: uniform on [0, 1] or similar
        # Let's try a few different initializations
        
        # Initialization 1: uniform on [0, 1]
        support_init1 = np.linspace(0, 1, n)
        weights_init1 = np.ones(n) / n
        
        # Initialization 2: geometric-like
        support_init2 = np.linspace(0, 10, n)
        weights_init2 = np.ones(n) / n
        
        # Initialization 3: concentrated near 0
        support_init3 = np.linspace(0, 1, n) ** 2
        weights_init3 = np.ones(n) / n
        
        inits = [
            (support_init1, weights_init1),
            (support_init2, weights_init2),
            (support_init3, weights_init3),
        ]
        
        for support_init, weights_init in inits:
            if time.time() > deadline - 20:
                break
            
            # Combine into single parameter vector
            params_init = np.concatenate([support_init, weights_init])
            
            # Bounds for differential evolution
            # Support: [0, 100] (we can scale later)
            # Weights: [0, 1]
            bounds = [(0, 100)] * n + [(0, 1)] * n
            
            # Run differential evolution with limited iterations
            # Use small population and few iterations for speed
            try:
                result = differential_evolution(
                    _neg_P,
                    bounds,
                    args=(n,),
                    maxiter=50,
                    popsize=15,
                    tol=1e-4,
                    seed=42,
                    polish=False,  # We'll do our own polish
                    update_interval=10
                )
            except Exception as e:
                continue
            
            if result.success or result.fun < -best_P:
                params_opt = result.x
                support_opt = np.maximum(params_opt[:n], 0.0)
                weights_opt = np.maximum(params_opt[n:], 1e-10)
                P_opt = _compute_P(support_opt, weights_opt)
                
                if P_opt > best_P:
                    best_P = P_opt
                    best_support = support_opt.copy()
                    best_weights = weights_opt.copy()
            
            # Also try local refinement from the DE result
            if time.time() > deadline - 15:
                break
            
            try:
                result_local = minimize(
                    _neg_P,
                    result.x,
                    args=(n,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                params_local = result_local.x
                support_local = np.maximum(params_local[:n], 0.0)
                weights_local = np.maximum(params_local[n:], 1e-10)
                P_local = _compute_P(support_local, weights_local)
                
                if P_local > best_P:
                    best_P = P_local
                    best_support = support_local.copy()
                    best_weights = weights_local.copy()
            except Exception as e:
                pass
        
        # Also try a direct SLSQP optimization from the baseline-like initialization
        if time.time() > deadline - 10:
            break
        
        # SLSQP with equality constraint that weights sum to 1
        def objective_slsqp(params):
            support = params[:n]
            weights = params[n:]
            support = np.maximum(support, 0.0)
            weights = np.maximum(weights, 1e-10)
            return -_compute_P(support, weights)
        
        def constraint_sum_weights(params):
            return np.sum(params[n:]) - 1.0
        
        constraints = [{'type': 'eq', 'fun': constraint_sum_weights}]
        
        # Try from uniform initialization
        support_slsqp_init = np.linspace(0, 5, n)
        weights_slsqp_init = np.ones(n) / n
        params_slsqp_init = np.concatenate([support_slsqp_init, weights_slsqp_init])
        
        bounds_slsqp = [(0, 50)] * n + [(0, 1)] * n
        
        try:
            result_slsqp = minimize(
                objective_slsqp,
                params_slsqp_init,
                method='SLSQP',
                bounds=bounds_slsqp,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if result_slsqp.fun < -best_P:
                params_slsqp = result_slsqp.x
                support_slsqp = np.maximum(params_slsqp[:n], 0.0)
                weights_slsqp = np.maximum(params_slsqp[n:], 1e-10)
                P_slsqp = _compute_P(support_slsqp, weights_slsqp)
                
                if P_slsqp > best_P:
                    best_P = P_slsqp
                    best_support = support_slsqp.copy()
                    best_weights = weights_slsqp.copy()
        except Exception as e:
            pass
    
    # Final cleanup: ensure valid output
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Normalize weights
    total_weight = np.sum(best_weights)
    if total_weight < 1e-9:
        # Fallback to baseline
        best_support = support_baseline
        best_weights = weights_baseline
    else:
        best_weights = best_weights / total_weight
    
    # Ensure no NaN or inf
    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        best_support = support_baseline
        best_weights = weights_baseline
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
