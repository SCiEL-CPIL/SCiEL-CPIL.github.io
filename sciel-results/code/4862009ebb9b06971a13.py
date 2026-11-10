import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    # Strategy: Global Grid-based Convex Optimization
    # We discretize the support into a large grid [0, L].
    # The objective P is linear in the weights w.
    # Constraints: sum(w) = 1, w >= 0.
    # This is a Linear Program (LP). We can solve it efficiently.
    # However, scipy.optimize.linprog might be slow for very large n or memory heavy.
    # Alternatively, since it's a simplex constraint and linear objective,
    # the optimal solution is a vertex of the simplex, meaning at most 1 non-zero weight?
    # Wait, P is bilinear in X, but linear in w for fixed support?
    # Let's check linearity.
    # P = sum_{k,l} p_k p_l P(Z < 2 s_l - s_k)
    # Z = X1 + X2. P(Z < t) = sum_{i,j} p_i p_j I(s_i + s_j < t).
    # So P = sum_{k,l,i,j} p_k p_l p_i p_j I(s_i + s_j < 2 s_l - s_k).
    # This is QUARTIC in p. It is NOT linear in p.
    # Therefore, it is a non-convex optimization problem over the simplex.
    # The "convex optimization" hint in the prompt might be a trap or refers to something else?
    # "maximizing the linear probability objective" -> This statement in the prompt's directive
    # "maximizing the linear probability objective" is likely INCORRECT for the actual metric P.
    # Let's re-read carefully: "Metric (higher is better): P = ... sum ... w_k * w_l * (sum of w_i * w_j ...)"
    # Yes, it's a product of 4 weights. It's a quartic form.
    # So simple LP won't work.
    
    # However, we can still use the "Grid" idea to allow for a much larger effective n
    # and use a more robust global optimization method like Simulated Annealing or
    # Multi-start Gradient Descent on the grid weights.
    
    # Given the time limit, a dense grid (e.g., 256 points) makes gradient descent
    # on weights feasible if we can compute gradients efficiently.
    # But computing the quartic form and its gradient for n=256 is O(n^4) if naive.
    # We can use the vectorized score computation from the current program which is O(n^2 log n) or similar.
    
    # Let's stick to the current program's efficient score calculation but apply it to a grid.
    # And use a global search method: Multi-start SLSQP or Nelder-Mead on the grid weights?
    # No, Nelder-Mead on 256 dims is bad.
    # SLSQP on 256 dims is okay if the function is smooth and we have a good start.
    
    # Alternative: Coordinate Descent on the grid.
    # Or, simpler: Use the current program's approach but with a larger n (e.g., 32 or 64)
    # and better initialization. The current program uses n=16.
    # The prompt suggests "discretizing the support into a large grid (e.g., 256 points)".
    # If we use n=256, the score computation cost is:
    # s_sums: 256x256 = 65k elements.
    # T: 256x256 = 65k elements.
    # searchsorted: O(n^2 log n) ~ 65k * 8 = 500k ops. Very fast.
    # SLSQP on 256 variables might be slow per iteration, but we can try.
    
    # Let's try n=128 or 256.
    # We will use a grid support: s_i = i * L / (n-1).
    # We optimize weights w on this fixed grid.
    
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
        
        # Thresholds T_{kl} = 2*s_l - s_k
        T = 2.0 * support[None, :] - support[:, None]
        
        # Count of elements in s_sorted < T[k, l]
        idx = np.searchsorted(s_sorted, T, side='left')
        
        # Zero-padded prefix sum
        p_prefix_pad = np.zeros(n * n + 1)
        p_prefix_pad[1:] = p_prefix
        
        # Prob(Z < T[k,l])
        prob_z_less = p_prefix_pad[idx]
        
        # Total prob = sum_{k, l} p_k * p_l * prob_z_less[k, l]
        total_prob = np.sum(p[:, None] * p[None, :] * prob_z_less)
        
        return total_prob

    def objective(weights, support):
        return -compute_score(support, weights)

    # Grid setup
    # Let's try a few grid sizes and pick the best.
    # n=64, 128, 256.
    # L (max support) should be large enough. In the current program, support was in [0,2].
    # Let's use L=4 or L=10 to be safe.
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    rng = np.random.default_rng(42)
    
    # Try different grid resolutions and max supports
    grid_configs = [
        (64, 2.0),
        (128, 4.0),
        (256, 8.0),
        (128, 1.0), # Maybe smaller support is better?
    ]
    
    for n, L in grid_configs:
        if time.time() > deadline - 60.0:
            break
            
        support = np.linspace(0, L, n)
        
        # Initial weights: Uniform? Or concentrated?
        # Let's try a few initializations.
        inits = [
            np.ones(n) / n,
            np.zeros(n), # Will be handled by optimizer? No, SLSQP needs valid start.
            # Dirac at some points?
        ]
        
        # Better init: Random sparse vectors?
        # SLSQP can handle this.
        
        for init_type in ['uniform', 'random']:
            if time.time() > deadline - 10.0:
                break
                
            if init_type == 'uniform':
                w0 = np.ones(n) / n
            else:
                w0 = rng.dirichlet(np.ones(n) * 0.1) # Sparse-ish
            
            # SLSQP
            bounds = [(0.0, 1.0) for _ in range(n)]
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
            ]
            
            try:
                res = minimize(
                    objective, 
                    w0, 
                    args=(support,), 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=constraints,
                    options={'maxiter': 50, 'ftol': 1e-8} # Fewer iterations for larger n
                )
                w_opt = res.x
                # Ensure validity
                w_opt = np.maximum(w_opt, 0)
                w_sum = np.sum(w_opt)
                if w_sum > 1e-9:
                    w_opt = w_opt / w_sum
                else:
                    continue
                    
                score = compute_score(support, w_opt)
                
                if score > best_score:
                    best_score = score
                    best_support = support.copy()
                    best_weights = w_opt.copy()
                    
            except Exception as e:
                # print(f"Optimization failed: {e}")
                pass
                
    # If grid optimization failed or didn't find anything, fall back to current logic?
    # Or just return the best found.
    # The current program achieves 1.40. Grid might find something better if the optimum
    # requires a more complex distribution than 16 points.
    
    if best_support is None:
        # Fallback to a simple known-good distribution if optimization failed
        # Use the result from the current program's logic roughly?
        # Let's just return a uniform distribution on [0,1] as a safe fallback
        n = 10
        best_support = np.linspace(0, 1, n)
        best_weights = np.ones(n) / n
        best_score = compute_score(best_support, best_weights)
        
    # Final cleanup
    best_support = np.maximum(best_support, 0)
    best_weights = np.maximum(best_weights, 0)
    w_sum = np.sum(best_weights)
    if w_sum > 1e-9:
        best_weights = best_weights / w_sum
    else:
        # Should not happen
        best_support = np.array([0.0])
        best_weights = np.array([1.0])

    best_support = best_support.flatten()
    best_weights = best_weights.flatten()
    
    return best_support, best_weights

# Import minimize
from scipy.optimize import minimize
