"""Packing 100 equal spheres in a unit cube: smooth-min gradient ascent + polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def _distances_and_radius(X):
    """Compute pairwise distances and the limiting radius for position array X (N,3)."""
    # Pairwise differences
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    # Distance to walls
    wall_dist = np.min(0.5 - np.abs(X), axis=1)
    min_wall = np.min(wall_dist)
    
    r = min(min_pair / 2.0, min_wall)
    return r, dist, wall_dist

def _soft_min_value(X, beta=50.0):
    """Smooth surrogate for the radius using log-sum-exp on squared distances."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    sq_dist = np.sum(diff * diff, axis=2)
    np.fill_diagonal(sq_dist, 1e30)
    
    # Soft min of distances: we want to maximize min(dist)
    # Use -log-sum-exp(-beta * dist) / beta as soft min
    dist = np.sqrt(sq_dist)
    np.fill_diagonal(dist, np.inf)
    
    # For walls
    wall = 0.5 - np.abs(X)
    min_wall_per_point = np.min(wall, axis=1)
    
    # Combine: soft min of all pair distances and wall distances
    # We'll use a simpler approach: penalize violations
    # Actually, let's use a differentiable objective: maximize t subject to
    # dist_ij >= 2t and wall_dist_i >= t
    # Use penalty method or just maximize a smooth approximation
    
    # Simple smooth max of min: use the fact that max t s.t. constraints
    # is equivalent to minimizing max violation
    # Let's use: f(X) = -log(sum(exp(-beta * dist_ij/2)) + exp(-beta * wall_i)) / beta
    # This approximates -min(dist/2, wall)
    
    pair_terms = np.exp(-beta * dist / 2.0)
    wall_terms = np.exp(-beta * min_wall_per_point)
    
    # Mask diagonal
    pair_terms = np.where(np.eye(N, dtype=bool), 0.0, pair_terms)
    
    log_sum = np.log(np.sum(pair_terms) + np.sum(wall_terms))
    return -log_sum / beta

def _objective(X):
    """Negative smooth radius to minimize."""
    return -_soft_min_value(X, beta=80.0)

def _objective_grad(X):
    """Gradient of negative smooth radius."""
    beta = 80.0
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1e10)
    
    min_wall = np.min(0.5 - np.abs(X), axis=1)
    
    pair_terms = np.exp(-beta * dist / 2.0)
    wall_terms = np.exp(-beta * min_wall)
    
    pair_terms = np.where(np.eye(N, dtype=bool), 0.0, pair_terms)
    
    sum_all = np.sum(pair_terms) + np.sum(wall_terms)
    if sum_all < 1e-300:
        return np.zeros_like(X)
    
    # d/dX of -log(sum)/beta = -1/(beta*sum) * d(sum)/dX
    # d(pair_terms_ij)/dX_i = pair_terms_ij * (-beta/2) * (X_i - X_j)/dist_ij
    # = -beta/2 * pair_terms_ij / dist_ij * (X_i - X_j)
    
    # For each pair (i,j), contribution to grad_i:
    # d(sum)/dX_i from pair terms: sum_j pair_terms_ij * (-beta/2) * (X_i - X_j)/dist_ij
    # = -beta/2 * sum_j [pair_terms_ij / dist_ij] * (X_i - X_j)
    
    # Weight matrix: W_ij = pair_terms_ij / dist_ij for i != j
    W = pair_terms / dist
    np.fill_diagonal(W, 0.0)
    
    # grad_pair_i = -beta/2 * sum_j W_ij * (X_i - X_j)
    # = -beta/2 * (sum_j W_ij * X_i - sum_j W_ij * X_j)
    # = -beta/2 * (row_sum_i * X_i - (W @ X)_i)
    
    row_sums = np.sum(W, axis=1)
    WX = W @ X
    
    grad_pair = -beta / 2.0 * (row_sums[:, np.newaxis] * X - WX)
    
    # Wall terms: d(wall_terms_i)/dX_i = wall_terms_i * (-beta) * d(min_wall_i)/dX_i
    # min_wall_i = min_k (0.5 - |X_ik|)
    # This is non-differentiable at the min, but we can use the closest wall
    # For simplicity, use the average gradient (subgradient)
    
    # d(0.5 - |x|)/dx = -sign(x) for x != 0
    # For the min, only the active constraint contributes. Use soft version:
    # Actually, let's just use the gradient of each wall term weighted
    
    wall_abs = np.abs(X)
    signs = np.sign(X)
    # d(0.5 - |x|)/dx = -sign(x)
    # But we need d(min_k (0.5-|x_k|))/dx_k which is -sign(x_k) for the active k
    
    # Simplified: use the gradient as if all walls contribute equally (soft)
    # d(wall_terms_i)/dX_ik = wall_terms_i * (-beta) * (-sign(X_ik)) = wall_terms_i * beta * sign(X_ik)
    # Wait: wall_i = min_k(0.5 - |X_ik|), so d(wall_terms_i)/dX_ik = wall_terms_i * d(wall_i)/dX_ik
    # d(wall_i)/dX_ik = -sign(X_ik) for the active constraint
    
    # Use soft min: weight each wall by softmax
    wall_vals = 0.5 - wall_abs
    # Soft min weights: w_k = exp(-beta * wall_vals_k) / sum_j exp(-beta * wall_vals_j)
    exp_wall = np.exp(-beta * wall_vals)
    exp_wall_sum = np.sum(exp_wall, axis=1, keepdims=True)
    wall_weights = exp_wall / exp_wall_sum  # (N, 3)
    
    # d(wall_i)/dX_ik = -sign(X_ik) * wall_weights_ik (soft version)
    grad_wall_i = -signs * wall_weights
    
    grad_wall = wall_terms[:, np.newaxis] * beta * grad_wall_i
    
    total_grad = (grad_pair + grad_wall) / sum_all
    
    # Objective is -log_sum/beta, so gradient is -total_grad/beta... wait
    # f = -log(sum)/beta
    # df/dX = -1/(beta*sum) * d(sum)/dX
    # d(sum)/dX = grad_pair_unscaled + grad_wall_unscaled where:
    # d(pair_sum)/dX_i = sum_j d(pair_terms_ij)/dX_i = sum_j pair_terms_ij * (-beta/2) * (X_i-X_j)/dist_ij
    # = -beta/2 * sum_j W_ij*(X_i-X_j) where W_ij = pair_terms_ij/dist_ij
    # So d(sum)/dX = grad_pair + grad_wall (as computed above, which already has the -beta/2 factor)
    # Wait, I computed grad_pair = -beta/2 * (row_sums*X - WX) which is d(pair_sum)/dX
    # And grad_wall = wall_terms * beta * grad_wall_i which is d(wall_sum)/dX
    # So d(sum)/dX = grad_pair + grad_wall
    # df/dX = -(grad_pair + grad_wall) / (beta * sum_all)
    
    grad_f = -(grad_pair + grad_wall) / (beta * sum_all)
    
    return grad_f

def _random_init(seed=42):
    """Generate a random valid initial configuration."""
    rng = np.random.RandomState(seed)
    # Start with a grid and add small random perturbations
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    # Add small random perturbations
    noise = rng.normal(0, 0.01, size=grid.shape)
    X = grid + noise
    
    # Clip to valid range
    X = np.clip(X, -0.45, 0.45)
    return X

def _polish(X, max_iter=500, tol=1e-12):
    """Polish the configuration using a more precise local optimization."""
    beta = 200.0
    
    def obj_and_grad(X):
        r, dist, wall_dist = _distances_and_radius(X)
        # Use a smooth penalty for violations
        # Actually, let's just use the soft min with high beta
        val = _soft_min_value(X, beta=beta)
        grad = _objective_grad(X)
        # Note: _objective_grad returns gradient of -soft_min, so we want negative
        return val, -grad
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * 3 * N
    
    best_X = X.copy()
    best_val = _soft_min_value(X, beta=beta)
    
    for attempt in range(3):
        res = minimize(
            lambda x: _soft_min_value(x, beta=beta),
            X,
            jac=lambda x: -_objective_grad(x),  # gradient of soft_min is -gradient of -soft_min
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10}
        )
        
        val = _soft_min_value(res.x, beta=beta)
        if val > best_val:
            best_val = val
            best_X = res.x.copy()
        
        # Perturb and try again
        if attempt < 2:
            X = best_X + np.random.normal(0, 0.001, size=best_X.shape)
            X = np.clip(X, -0.49, 0.49)
    
    return best_X

def construct():
    start_time = time.time()
    
    # Generate multiple random initializations
    best_X = None
    best_r = -1.0
    
    # Try several seeds
    for seed in range(10):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        X = _random_init(seed=seed)
        r_init, _, _ = _distances_and_radius(X)
        
        # Gradient ascent using L-BFGS-B
        bounds = [(-0.5, 0.5)] * 3 * N
        
        try:
            res = minimize(
                _objective,
                X,
                jac=_objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            r_res, _, _ = _distances_and_radius(res.x)
            if r_res > best_r:
                best_r = r_res
                best_X = res.x.copy()
        except Exception:
            continue
    
    # Polish the best result
    if best_X is not None and time.time() - start_time < DEADLINE - 5:
        best_X = _polish(best_X, max_iter=200)
        r_final, _, _ = _distances_and_radius(best_X)
        best_r = r_final
    
    # Final validation and clipping
    if best_X is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_X = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    # Ensure all points are strictly inside
    best_X = np.clip(best_X, -0.499, 0.499)
    
    return best_X
