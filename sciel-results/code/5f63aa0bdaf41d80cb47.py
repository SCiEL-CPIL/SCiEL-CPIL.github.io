# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _objective(params, n):
    """Objective function to minimize (negative of P)."""
    # params: first n are support points (log-spaced or linear), next n are log-weights
    # We'll use a transformation to ensure support >= 0 and weights > 0
    
    # Split params
    s_params = params[:n]
    w_params = params[n:]
    
    # Support: use exp to ensure positive, but we want to allow 0.
    # Let's use softplus or just exp(s_params) * scale + offset?
    # Actually, the optimal support likely has a point at 0.
    # Let's use: support[i] = exp(s_params[i])
    # But if we want 0, exp can't be 0.
    # Alternative: support[i] = s_params[i] if s_params[i] > 0 else 0? Non-differentiable.
    # Better: support[i] = np.maximum(0, s_params[i])? Non-differentiable.
    # Let's use support[i] = exp(s_params[i]). The minimum value will be small but > 0.
    # Or, we can fix one point to be 0 and optimize the rest.
    # Let's try: support[0] = 0, support[i] = exp(s_params[i-1]) for i>0.
    
    support = np.zeros(n, dtype=np.float64)
    support[1:] = np.exp(s_params[1:])
    
    # Weights: use softmax to ensure they sum to 1 and are positive
    weights = np.exp(w_params - np.max(w_params))
    weights = weights / np.sum(weights)
    
    # Calculate P
    # P = sum_{k,l} w_k w_l * (sum_{i,j} w_i w_j [s_i + s_j < 2*s_l - s_k])
    # Let A = sum of weights for pairs (i,j) such that s_i + s_j < threshold
    # This is equivalent to:
    # P = sum_{k,l,i,j} w_k w_l w_i w_j [s_i + s_j < 2*s_l - s_k]
    
    # Vectorized computation:
    # s_i + s_j < 2*s_l - s_k
    # s_i + s_k < 2*s_l - s_j  (symmetry doesn't help directly)
    
    # Let's compute all pairwise sums S_ij = s_i + s_j
    S = support[:, np.newaxis] + support[np.newaxis, :]
    
    # Threshold T_kl = 2*s_l - s_k
    # We need to sum w_i w_j for all (i,j) where S_ij < T_kl
    # This is a cumulative sum problem.
    
    # Sort support and S?
    # Since S is not sorted, we can't just use cumulative sum directly on S.
    # But we can sort the values of S and use searchsorted.
    
    # Flatten S
    S_flat = S.flatten()
    
    # We need for each (k,l), the sum of w_i w_j for S_ij < T_kl.
    # Let V_ij = w_i w_j.
    V = weights[:, np.newaxis] * weights[np.newaxis, :]
    V_flat = V.flatten()
    
    # Sort S_flat and V_flat together
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    V_sorted = V_flat[sort_idx]
    
    # Cumulative sum of V_sorted
    V_cumsum = np.cumsum(V_sorted)
    
    # Now for each (k,l), compute T_kl
    T = 2 * support[np.newaxis, :] - support[:, np.newaxis]
    T_flat = T.flatten()
    
    # Find indices in S_sorted where S_sorted < T_flat
    # np.searchsorted(S_sorted, T_flat, side='left') gives the index of the first element >= T_flat
    # So the number of elements < T_flat is idx
    idx = np.searchsorted(S_sorted, T_flat, side='left')
    
    # Sum of V_sorted[:idx]
    # We can use V_cumsum[idx] but careful with idx=0
    # Create a padded cumsum array
    V_cumsum_padded = np.concatenate(([0.0], V_cumsum))
    P_inner = V_cumsum_padded[idx]
    
    # Now P = sum_{k,l} w_k w_l * P_inner[k,l]
    W_prod = weights[:, np.newaxis] * weights[np.newaxis, :]
    P = np.sum(W_prod * P_inner.reshape(n, n))
    
    # Return negative P to minimize
    return -P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds
    
    # Start with a reasonable initial guess
    # The baseline is uniform on {0, ..., 31}.
    # Let's try a geometric distribution or similar.
    # Or just optimize from the baseline.
    
    best_support = None
    best_weights = None
    best_score = -1.0
    
    # Try different n values
    # n=2 is fast, n=4, n=8, n=16...
    # Let's start with small n and increase if time permits.
    
    # n=2
    # support = [0, x], weights = [p, 1-p]
    # P = ?
    # If support = [0, 1], weights = [0.5, 0.5]
    # S = [[0,1],[1,2]]
    # T_kl = 2*l - k
    # k=0,l=0: T=0. S<0: none. P_inner=0.
    # k=0,l=1: T=2. S<2: 0,1,1. P_inner = 0.25+0.25+0.25 = 0.75.
    # k=1,l=0: T=-1. S<-1: none. P_inner=0.
    # k=1,l=1: T=1. S<1: 0. P_inner=0.25.
    # P = 0.25*0 + 0.25*0.75 + 0.25*0 + 0.25*0.25 = 0.1875 + 0.0625 = 0.25.
    
    # Let's use optimization for n=4, 8, 16.
    
    for n in [4, 8, 16]:
        if time.time() > deadline - 60:  # Leave 60s margin
            break
            
        # Initial guess: log-spaced support, uniform weights
        s_init = np.zeros(n)
        s_init[1:] = np.log(np.linspace(1, n, n-1))  # exp(log(x)) = x
        w_init = np.zeros(n)
        
        params_init = np.concatenate([s_init, w_init])
        
        # Bounds
        # s_params: support = exp(s_params). Let's allow s_params in [-10, 10]
        # w_params: softmax. Let's allow w_params in [-10, 10]
        bounds = [(-10, 10)] * (2 * n)
        
        # Use Powell first (faster, local)
        res = minimize(_objective, params_init, args=(n,), method='Powell', 
                       options={'maxiter': 100, 'disp': False})
        
        if res.success:
            params_opt = res.x
            s_opt = params_opt[:n]
            w_opt = params_opt[n:]
            
            support = np.zeros(n)
            support[1:] = np.exp(s_opt[1:])
            weights = np.exp(w_opt - np.max(w_opt))
            weights = weights / np.sum(weights)
            
            # Verify score
            score = -_objective(params_opt, n)
            if score > best_score:
                best_score = score
                best_support = support
                best_weights = weights
        
        # Use Differential Evolution (global, slower) if time permits
        if time.time() > deadline - 120:
            continue
            
        # DE bounds
        de_bounds = [(-5, 5)] * (2 * n)
        
        res_de = differential_evolution(_objective, de_bounds, args=(n,), 
                                         maxiter=50, popsize=10, tol=1e-6, 
                                         seed=42, polish=False)
        
        if res_de.success:
            params_de = res_de.x
            s_de = params_de[:n]
            w_de = params_de[n:]
            
            support_de = np.zeros(n)
            support_de[1:] = np.exp(s_de[1:])
            weights_de = np.exp(w_de - np.max(w_de))
            weights_de = weights_de / np.sum(weights_de)
            
            score_de = -_objective(params_de, n)
            if score_de > best_score:
                best_score = score_de
                best_support = support_de
                best_weights = weights_de
    
    # If optimization failed or returned None, use baseline
    if best_support is None:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    else:
        # Normalize weights to sum to 1 (though they should already be)
        best_weights = best_weights / np.sum(best_weights)
        # Ensure support is non-negative
        best_support = np.maximum(0, best_support)
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
