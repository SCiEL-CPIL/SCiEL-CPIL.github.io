# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def _calc_P(support, weights):
    """
    Calculate P[X1 + X2 + X3 < 2*X4] for a discrete distribution.
    support: np.array of shape (n,)
    weights: np.array of shape (n,), assumed normalized
    """
    s = support
    w = weights
    n = len(s)
    
    # Precompute pairwise sums and weights
    # Sums[i, j] = s[i] + s[j]
    # Weights[i, j] = w[i] * w[j]
    sums = s[:, None] + s[None, :]
    w_prod = w[:, None] * w[None, :]
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            # Sum w_i * w_j where s_i + s_j < threshold
            inner_sum = np.sum(w_prod[sums < threshold])
            P += w[k] * w[l] * inner_sum
            
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    best_P = 0.0
    best_support = np.array([0.0])
    best_weights = np.array([1.0])
    
    # Try small n values
    # n=2 is very fast, n=3, 4, 5 are also fast.
    # Optimization cost is dominated by the number of iterations and function evaluations.
    # _calc_P is O(n^2) for the loop over k,l, and each step is O(n^2) for the sum.
    # So _calc_P is O(n^4). For n=6, 6^4 = 1296 ops. Very fast.
    # We can do many restarts.
    
    for n in [2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
            
        support = np.arange(n, dtype=np.float64)
        
        # Number of restarts depends on n and remaining time
        # For small n, we can afford more restarts.
        num_restarts = 50 if n <= 4 else 20
        
        for _ in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Random initial weights, normalized
            w_init = np.random.dirichlet(np.ones(n))
            
            def objective(w):
                # Ensure non-negativity and normalization for safety, 
                # though optimizer should handle constraints.
                # L-BFGS-B handles bounds. We use bounds=[0,1].
                # Sum constraint is handled by SLSQP or by parameterizing w_i = exp(x_i) / sum(exp(x_j)).
                # Let's use SLSQP for equality constraint sum(w)=1 and bounds w>=0.
                if np.any(w < -1e-9):
                    return 1e6
                w_norm = w / np.sum(w)
                return -_calc_P(support, w_norm)
            
            def constraint_sum(w):
                return np.sum(w) - 1.0
            
            constraints = [{'type': 'eq', 'fun': constraint_sum}]
            bounds = [(0, 1)] * n
            
            try:
                res = so.minimize(
                    objective, 
                    w_init, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                if res.success:
                    w_opt = res.x
                    w_opt = np.clip(w_opt, 0, None)
                    w_opt = w_opt / np.sum(w_opt)
                    P_val = _calc_P(support, w_opt)
                    
                    if P_val > best_P:
                        best_P = P_val
                        best_support = support.copy()
                        best_weights = w_opt.copy()
                        
            except Exception:
                pass
                
    # If no optimization succeeded, fall back to seed
    if best_P == 0.0:
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
        
    return best_support, best_weights

# EVOLVE-BLOCK-END
