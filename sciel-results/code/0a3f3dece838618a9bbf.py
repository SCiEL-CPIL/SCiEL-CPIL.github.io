import numpy as np
import time
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin: 630s limit, use 600s for computation

    def compute_score(support, weights):
        n = len(support)
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        p = weights / w_sum
        
        # Pairwise sums and weights for Z = X1 + X2
        s_sums = support[:, None] + support[None, :]
        p_prods = p[:, None] * p[None, :]
        
        # Flatten and sort
        s_sums_flat = s_sums.flatten()
        p_prods_flat = p_prods.flatten()
        
        sort_idx = np.argsort(s_sums_flat)
        s_sorted = s_sums_flat[sort_idx]
        p_sorted = p_prods_flat[sort_idx]
        
        # Prefix sums of probabilities for Z = X1 + X2
        p_prefix = np.cumsum(p_sorted)
        
        # Vectorized calculation for X3 and X4
        # We need sum over k, l of p_k * p_l * P(Z < 2*s_l - s_k)
        # T[k, l] = 2 * s_l - s_k
        T = 2.0 * support[None, :] - support[:, None]
        
        # idx[k, l] is the count of elements in s_sorted < T[k, l]
        idx = np.searchsorted(s_sorted, T, side='left')
        
        # Zero-padded prefix sum
        p_prefix_pad = np.zeros(n * n + 1)
        p_prefix_pad[1:] = p_prefix
        
        prob_z_less = p_prefix_pad[idx]
        
        total_prob = np.sum(p[:, None] * p[None, :] * prob_z_less)
        
        return total_prob

    def negative_score_log(weights_log):
        """
        Objective function for scipy.optimize.minimize.
        Uses log-weights to enforce positivity naturally.
        weights = exp(weights_log) / sum(exp(weights_log))
        """
        # Convert log-weights to probabilities
        # Subtract max for numerical stability
        w_max = np.max(weights_log)
        exp_w = np.exp(weights_log - w_max)
        p = exp_w / np.sum(exp_w)
        
        score = compute_score(support_grid, p)
        return -score

    # Fixed support grid
    # Using a dense grid on [0, 1] is a reasonable heuristic for this type of problem.
    # The optimal support often lies in a bounded region.
    n_grid = 256
    support_grid = np.linspace(0, 1, n_grid)
    
    # Initial weights: uniform
    initial_weights_log = np.zeros(n_grid)
    
    # Optimization options
    # BFGS is a quasi-Newton method that approximates the Hessian, good for smooth objectives.
    # Since the objective is differentiable (piecewise linear with respect to weights, 
    # but the sorting creates kinks, however BFGS often handles these well in practice 
    # or we can rely on the fact that the probability mass is spread out enough).
    
    # Actually, compute_score involves sorting and searchsorted, which makes the function 
    # piecewise constant/linear in weights, leading to non-smoothness. 
    # However, with a dense grid, the kinks are very fine. 
    # Let's try L-BFGS-B or BFGS. 
    # To handle the simplex constraint robustly, we optimize over log-weights (unconstrained) 
    # which maps to the interior of the simplex.
    
    best_score = -np.inf
    best_weights = None
    
    # Run optimization
    # We might need multiple restarts or different initializations if it gets stuck, 
    # but with a fixed grid and convex-like behavior (probability is linear in p for fixed support), 
    # the problem over the simplex is convex maximization? No, maximizing a linear function 
    # over a simplex is easy (vertex), but here the "coefficients" depend on p? 
    # Wait, P = sum_{k,l} p_k p_l P(Z < 2s_l - s_k). 
    # This is quadratic in p! It's not linear. 
    # So it's a Quadratic Programming problem (maximize p^T A p subject to sum p = 1, p >= 0).
    # A is symmetric? A_{kl} = P(Z < 2s_l - s_k). 
    # Is A symmetric? P(Z < 2s_l - s_k) vs P(Z < 2s_k - s_l). Not obviously symmetric.
    # However, we can use scipy.optimize.minimize on the log-weights.
    
    try:
        # Time budget for optimization
        # scipy.optimize.minimize doesn't have a direct time limit, but we can monitor.
        # For now, we run it and hope it converges fast. 
        # n_grid=256 means 256 variables. BFGS should be fine.
        
        # Use a callback to check time? 
        # Simpler: just run it. If it's too slow, we reduce n_grid.
        
        res = minimize(
            negative_score_log, 
            initial_weights_log, 
            method='BFGS', 
            options={'maxiter': 1000, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < -best_score:
            # Convert back to weights
            w_max = np.max(res.x)
            exp_w = np.exp(res.x - w_max)
            best_weights = exp_w / np.sum(exp_w)
            best_score = -res.fun
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to uniform
        best_weights = np.ones(n_grid) / n_grid
        best_score = compute_score(support_grid, best_weights)
    
    # If time remains, we can try to refine with a smaller grid or different grid range
    # But 256 points on [0,1] is a strong baseline.
    
    # Check time
    if time.time() > deadline - 5:
        pass # Return current best
    
    # Final cleanup
    best_support = support_grid.flatten()
    best_weights = np.maximum(best_weights, 0).flatten()
    
    # Ensure weights sum to 1 (already done, but just in case)
    w_sum = np.sum(best_weights)
    if w_sum > 1e-9:
        best_weights = best_weights / w_sum
    else:
        best_weights = np.ones(n_grid) / n_grid
        best_support = support_grid.flatten()
        
    return best_support, best_weights
