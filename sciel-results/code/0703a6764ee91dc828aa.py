# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for discrete measure with given support and weights.
    Uses O(n^2 log n) vectorized approach with sorted pairwise sums and cumulative weights.
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    w_sum = w.sum()
    if w_sum <= 1e-15:
        return 0.0
    w = w / w_sum
    
    # Pairwise sums and weights for X2+X3
    s_pair = s[:, None] + s[None, :]  # (n, n)
    w_pair = w[:, None] * w[None, :]  # (n, n)
    
    # Flatten and sort by sum
    flat_s = s_pair.ravel()  # (n*n,)
    flat_w = w_pair.ravel()  # (n*n,)
    
    order = np.argsort(flat_s, kind='mergesort')
    sorted_s = flat_s[order]
    sorted_w = flat_w[order]
    cum_w = np.cumsum(sorted_w)
    
    # Targets: targets[k, l] = 2*s[l] - s[k]
    targets = 2.0 * s[None, :] - s[:, None]  # (n, n), targets[k, l]
    
    # For each target, find cumulative weight of pairs with sum < target
    flat_targets = targets.ravel()  # (n*n,)
    idx = np.searchsorted(sorted_s, flat_targets, side='left')  # (n*n,)
    
    P_X2X3 = np.zeros_like(flat_targets)
    mask = idx > 0
    P_X2X3[mask] = cum_w[idx[mask] - 1]
    
    P_X2X3 = P_X2X3.reshape(n, n)
    
    # P = sum_k sum_l w[k]*w[l] * P_X2X3[k, l]
    P = np.sum(w[:, None] * w[None, :] * P_X2X3)
    
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1..X4 are i.i.d. with law mu.
    
    Strategy:
    1. Discretize support on a logarithmic grid to control separation ratios.
    2. Combinatorial search for support subsets (packing).
    3. For each subset, optimize weights using CCP (Concave-Convex Procedure) 
       to handle the non-convexity of the probability landscape.
    """
    start_time = time.time()
    deadline = start_time + 610.0  # safety margin before 630s kill

    best_P = -1.0
    best_support = None
    best_weights = None

    def update_best(s, w):
        nonlocal best_P, best_support, best_weights
        if len(s) == 0:
            return
        w = np.asarray(w, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        w_sum = w.sum()
        if w_sum <= 1e-15:
            return
        w = w / w_sum
        # Ensure non-negative support
        if np.any(s < -1e-12):
            return
        # Ensure finite
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
            return
        P = _compute_P(s, w)
        if P > best_P:
            best_P = P
            best_support = s.copy()
            best_weights = w.copy()

    # --- Phase 1: Analytic 2-point baseline ---
    def two_point_P(p):
        """P[X1+X2+X3 < 2*X4] for 2-point {0, 1} with mass p on 0."""
        return (1.0 - p) * (p**3 + 3.0 * p**2 * (1.0 - p))
    
    a, b = 0.01, 0.99
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = two_point_P(c)
    fd = two_point_P(d)
    for _ in range(100):
        if fc < fd:
            a = c; c = d; fc = fd
            d = a + gr * (b - a); fd = two_point_P(d)
        else:
            b = d; d = c; fd = fc
            c = b - gr * (b - a); fc = two_point_P(c)
    p_opt = (a + b) / 2.0
    update_best(np.array([0.0, 1.0]), np.array([p_opt, 1.0 - p_opt]))

    # --- Phase 2: Log-Grid Combinatorial Search + CCP ---
    
    # Define a logarithmic grid for support points
    # We want points that are well-separated to create distinct phase boundaries
    # in the indicator function.
    log_min = np.log(1e-3)
    log_max = np.log(1e3)
    n_grid = 30
    log_grid = np.linspace(log_min, log_max, n_grid)
    grid_points = np.exp(log_grid)
    
    # Pre-compute pairwise sums and targets for all grid points?
    # No, n_grid is 30, so n_grid^2 = 900.
    # But we are selecting subsets of size k (e.g., 3, 4, 5).
    # We will iterate over combinations of grid points.
    
    # Helper: Optimize weights for a fixed support using CCP
    def optimize_weights_ccp(support, max_iter=50, tol=1e-8):
        n = len(support)
        # Initialize with uniform weights
        w = np.ones(n) / n
        
        for _ in range(max_iter):
            # Compute current P and gradients?
            # P(w) = sum_{k,l} w_k w_l C_{kl} where C_{kl} = P[X2+X3 < 2s_l - s_k]
            # This is a quadratic form w^T C w.
            # C is symmetric? C_{kl} depends on s_k, s_l.
            # Let's compute the matrix C for this support.
            
            s = support
            # Pairwise sums for X2+X3
            s_pair = s[:, None] + s[None, :]
            flat_s = s_pair.ravel()
            order = np.argsort(flat_s, kind='mergesort')
            sorted_s = flat_s[order]
            
            # Targets
            targets = 2.0 * s[None, :] - s[:, None]
            flat_targets = targets.ravel()
            
            # For each target, count/weight of pairs with sum < target
            # Since we are optimizing weights, the weights of the pairs are w_i w_j.
            # C_{kl} = sum_{i,j: s_i+s_j < 2s_l-s_k} w_i w_j
            # This is linear in the vector v = w \otimes w (Kronecker product).
            # Let V = vec(w w^T). Then C_vec = M @ V where M is an indicator matrix.
            # P = sum_{k,l} w_k w_l C_{kl} = V @ C_vec?
            # Actually P = w^T (C_matrix) w.
            # C_matrix_{kl} = sum_{i,j} I(s_i+s_j < 2s_l-s_k) w_i w_j.
            # This is not constant! It depends on w.
            # So P(w) is a quartic form? No.
            # P = sum_{k,l} w_k w_l [ sum_{i,j} I(...) w_i w_j ]
            # P = sum_{k,l,i,j} I(s_i+s_j < 2s_l-s_k) w_k w_l w_i w_j
            # This is a degree 4 polynomial in w.
            
            # Gradient:
            # dP/dw_m = sum_{k,l,i,j} I(...) [delta_{km} w_l w_i w_j + w_k delta_{lm} w_i w_j + w_k w_l delta_{im} w_j + w_k w_l w_i delta_{jm}]
            
            # To use CCP, we approximate the non-convex parts.
            # Or simpler: use scipy.optimize.minimize with numerical gradient?
            # Given n is small (3-5), numerical gradient is fine.
            
            def neg_P(w_vec):
                if np.any(w_vec < 0) or abs(w_vec.sum() - 1.0) > 1e-6:
                    return 1e10
                return -_compute_P(support, w_vec)
            
            # Use L-BFGS-B with bounds
            bounds = [(0, 1)] * n
            # Constraint: sum(w) = 1 is handled by L-BFGS-B if we use a constrained solver?
            # No, L-BFGS-B doesn't handle equality constraints.
            # We can parameterize w via softmax or just optimize n-1 variables.
            # Let's optimize n-1 variables, w_n = 1 - sum(w_1..w_{n-1}).
            
            def neg_P_reduced(x):
                w_full = np.zeros(n)
                w_full[:-1] = x
                w_full[-1] = 1.0 - np.sum(x)
                if np.any(w_full < -1e-9):
                    return 1e10
                w_clipped = np.maximum(w_full, 0)
                w_norm = w_clipped / w_clipped.sum()
                return -_compute_P(support, w_norm)
            
            x0 = np.ones(n-1) / n
            try:
                res = minimize(neg_P_reduced, x0, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-12})
                w_opt = np.zeros(n)
                w_opt[:-1] = res.x
                w_opt[-1] = 1.0 - np.sum(res.x)
                w_opt = np.maximum(w_opt, 0)
                w_opt = w_opt / w_opt.sum()
                
                P_opt = -res.fun
                if P_opt > _compute_P(support, w):
                    w = w_opt
                else:
                    break
            except Exception:
                pass
                
        return w

    # Search over subsets of grid points
    # We try sizes 3, 4, 5
    # To avoid combinatorial explosion, we use a greedy/heuristic approach:
    # Start with the 2-point best, then try adding points from the grid.
    
    current_support = best_support.copy()
    current_weights = best_weights.copy()
    
    # Sort grid points
    grid_points = np.sort(grid_points)
    
    # We will try to add points one by one, choosing the one that maximizes P
    # after weight optimization.
    
    n_max_points = 6
    current_n = len(current_support)
    
    while current_n < n_max_points:
        if time.time() > deadline - 60.0:
            break
            
        best_candidate_support = None
        best_candidate_weights = None
        best_candidate_P = _compute_P(current_support, current_weights)
        
        # Candidates: all grid points not already in support
        # Also try perturbed versions? No, stick to grid for now.
        candidates = []
        for gp in grid_points:
            # Check if gp is too close to existing points
            min_dist = np.min(np.abs(current_support - gp))
            if min_dist > 1e-6:
                candidates.append(gp)
        
        # Limit candidates to top 20 most distinct? Or just iterate all.
        # n_grid is 30, so at most 30 candidates.
        
        for gp in candidates:
            if time.time() > deadline - 30.0:
                break
                
            new_support = np.sort(np.concatenate([current_support, [gp]]))
            new_weights = np.zeros(len(new_support))
            # Initialize weights: keep old weights proportional, give new point small weight
            # Or just uniform?
            new_weights[:] = 1.0 / len(new_support)
            
            # Optimize weights
            w_opt = optimize_weights_ccp(new_support)
            
            P_new = _compute_P(new_support, w_opt)
            
            if P_new > best_candidate_P:
                best_candidate_P = P_new
                best_candidate_support = new_support
                best_candidate_weights = w_opt
        
        if best_candidate_support is not None:
            update_best(best_candidate_support, best_candidate_weights)
            current_support = best_candidate_support
            current_weights = best_candidate_weights
            current_n = len(current_support)
        else:
            break

    # --- Phase 3: Local Refinement of Support (Nelder-Mead) ---
    # Now that we have a good discrete structure, refine the support positions
    # to maximize P.
    
    if time.time() < deadline - 20.0 and best_support is not None:
        n_best = len(best_support)
        
        def _neg_P_full(params):
            # params: [s_1, ..., s_n, log_w_1, ..., log_w_n]
            s = params[:n_best]
            log_w = params[n_best:]
            
            if np.any(s < -1e-9):
                return 1e10
            
            # Softmax for weights
            log_w_shifted = log_w - log_w.max()
            w = np.exp(log_w_shifted)
            w = w / w.sum()
            
            # Sort support and weights
            order = np.argsort(s)
            s_sorted = s[order]
            w_sorted = w[order]
            
            return -_compute_P(s_sorted, w_sorted)
        
        # Initial params
        s_init = best_support
        w_init = best_weights
        log_w_init = np.log(w_init + 1e-300)
        params0 = np.concatenate([s_init, log_w_init])
        
        try:
            result = minimize(_neg_P_full, params0, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            s_opt = result.x[:n_best]
            log_w_opt = result.x[n_best:]
            
            if np.any(s_opt < -1e-9):
                s_opt = np.maximum(s_opt, 0)
            
            log_w_shifted = log_w_opt - log_w_opt.max()
            w_opt = np.exp(log_w_shifted)
            w_opt = w_opt / w_opt.sum()
            
            order = np.argsort(s_opt)
            s_sorted = s_opt[order]
            w_sorted = w_opt[order]
            
            update_best(s_sorted, w_sorted)
        except Exception:
            pass

    # --- Final Output ---
    if best_support is None:
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
    
    # Clean up: remove points with negligible weight
    mask = best_weights > 1e-10
    best_support = best_support[mask]
    best_weights = best_weights[mask]
    
    # Normalize
    best_weights = best_weights / best_weights.sum()
    
    # Validation
    assert len(best_support) == len(best_weights)
    assert len(best_support) >= 1
    assert np.all(np.isfinite(best_support))
    assert np.all(np.isfinite(best_weights))
    assert np.all(best_support >= -1e-9)
    assert np.all(best_weights >= -1e-9)
    assert np.sum(best_weights) > 1e-9
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)

# EVOLVE-BLOCK-END
