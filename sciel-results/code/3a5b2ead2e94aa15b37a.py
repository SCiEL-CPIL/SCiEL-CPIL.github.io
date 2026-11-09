# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin
    
    def _compute_P(support, weights):
        """
        Compute P[X1+X2+X3 < 2*X4] efficiently.
        support: (n,)
        weights: (n,)
        """
        n = len(support)
        if n == 0:
            return 0.0
        
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            return 0.0
            
        w = weights / w_sum
        s = support
        
        # Pairwise sums s_i + s_j
        # We need to sum w_k w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i w_j)
        # Let S be the n x n matrix of sums s_i + s_j
        # Let W be the n x n matrix of weights w_i w_j
        # For each pair (k, l), we look at threshold T = 2*s_l - s_k
        # The inner sum is the sum of W[i,j] where S[i,j] < T
        
        S = s[:, np.newaxis] + s[np.newaxis, :]
        W = w[:, np.newaxis] * w[np.newaxis, :]
        
        # Flatten S and W for sorting
        flat_S = S.ravel()
        flat_W = W.ravel()
        
        # Sort S and keep corresponding W
        sort_idx = np.argsort(flat_S, kind='stable')
        sorted_S = flat_S[sort_idx]
        sorted_W = flat_W[sort_idx]
        cum_W = np.cumsum(sorted_W)
        
        P = 0.0
        # Iterate over all (k, l) pairs
        # k is index for s_k, l is index for s_l
        for k in range(n):
            for l in range(n):
                threshold = 2.0 * s[l] - s[k]
                # Find number of elements < threshold
                idx = np.searchsorted(sorted_S, threshold, side='left')
                inner_sum = cum_W[idx - 1] if idx > 0 else 0.0
                P += w[k] * w[l] * inner_sum
                
        return P

    def _objective(params, n):
        """
        Objective function for optimization.
        params: [s_1, ..., s_{n-1}, alpha_0, ..., alpha_{n-1}]
        s_0 is fixed to 0.
        s_i for i>0 are sorted.
        weights are softmax(alpha).
        """
        # Extract support
        s = np.empty(n)
        s[0] = 0.0
        if n > 1:
            s[1:] = params[:n-1]
            # Sort to break symmetry
            s[1:] = np.sort(s[1:])
        
        # Extract weights
        alphas = params[n-1:]
        # Softmax for stable weights
        max_alpha = np.max(alphas)
        exp_alphas = np.exp(alphas - max_alpha)
        weights = exp_alphas / np.sum(exp_alphas)
        
        P = _compute_P(s, weights)
        return -P

    best_P = -1.0
    best_support = None
    best_weights = None
    
    # Try different numbers of points
    for n in [2, 3, 4, 5, 6]:
        if time.time() > deadline - 30:
            break
            
        # Define bounds
        # Support points s_1..s_{n-1} in [0, 100]
        # Alphas in [-10, 10]
        bounds = [(0.0, 100.0)] * (n - 1) + [(-10.0, 10.0)] * n
        
        # Run Differential Evolution
        # Use a reasonable population size and maxiter
        # Time limit is handled by the outer loop, but we can also use 'maxiter'
        # DE can be slow, so we cap it.
        
        try:
            res = differential_evolution(
                lambda p: _objective(p, n),
                bounds,
                seed=42,
                maxiter=100,  # Keep it lowish to allow multiple n
                popsize=15,
                tol=1e-6,
                atol=1e-8,
                polish=True  # Use L-BFGS-B polish inside DE? Might be slow. Let's do manual polish.
            )
            
            if res.success or res.fun < -best_P:
                P_val = -res.fun
                if P_val > best_P:
                    # Reconstruct support and weights
                    s = np.empty(n)
                    s[0] = 0.0
                    if n > 1:
                        s[1:] = np.sort(res.x[:n-1])
                    
                    alphas = res.x[n-1:]
                    max_alpha = np.max(alphas)
                    exp_alphas = np.exp(alphas - max_alpha)
                    weights = exp_alphas / np.sum(exp_alphas)
                    
                    best_P = P_val
                    best_support = s.copy()
                    best_weights = weights.copy()
                    
        except Exception as e:
            pass
        
        # Local Polish with Nelder-Mead on the best found so far for this n
        if best_support is not None and time.time() < deadline - 20:
            try:
                # Initial params for polish
                init_params = np.concatenate([best_support[1:], np.log(best_weights + 1e-15)])
                
                # Nelder-Mead doesn't use bounds, so we rely on the landscape
                # But it might go out of bounds. We clamp in the objective? 
                # The objective function _objective doesn't clamp s to [0, 100].
                # If s goes negative, it's invalid for the task (support < -1e-9 is invalid).
                # We should ensure the returned support is valid.
                
                res_polish = minimize(
                    lambda p: _objective(p, n),
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if res_polish.fun < -best_P:
                    P_val = -res_polish.fun
                    if P_val > best_P:
                        s = np.empty(n)
                        s[0] = 0.0
                        if n > 1:
                            s[1:] = np.sort(res_polish.x[:n-1])
                        
                        # Ensure support is non-negative
                        s = np.maximum(s, 0.0)
                        
                        alphas = res_polish.x[n-1:]
                        max_alpha = np.max(alphas)
                        exp_alphas = np.exp(alphas - max_alpha)
                        weights = exp_alphas / np.sum(exp_alphas)
                        
                        best_P = P_val
                        best_support = s.copy()
                        best_weights = weights.copy()
                        
            except Exception:
                pass

    # Final Validation
    if best_support is None:
        # Fallback
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
        
    # Ensure validity: support >= 0, weights > 0
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 1e-15)
    
    # Normalize weights? The evaluator does it, but returning normalized is cleaner.
    # The prompt says: "w_i = weights[i] / sum(weights)". So raw weights are fine.
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
