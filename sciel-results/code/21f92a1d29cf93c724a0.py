import numpy as np
from scipy.optimize import minimize
import time

N = 32

def _compute_penalty_and_grad(flat_x, mu=500.0, eps=1e-8):
    """
    Penalty-based objective with analytical gradient.
    Maximizes sum of radii where radius_i = min(dist_to_boundary_i, min_j dist_ij/2)
    Uses soft penalty for constraint violations to make it differentiable.
    """
    x = flat_x.reshape(N, 2)
    
    # Distances to boundaries for each circle
    dist_bound = np.minimum(np.minimum(x[:, 0], 1 - x[:, 0]), np.minimum(x[:, 1], 1 - x[:, 1]))
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + eps)
    np.fill_diagonal(dists, np.inf)
    
    # For each circle i, the radius is limited by:
    # - distance to boundary
    # - half distance to every other circle j
    # r_i = min(dist_bound_i, min_j dists[i,j]/2)
    
    # Compute the "effective radius" for each circle
    # r_i = min over all constraints
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    r_raw = np.minimum(dist_bound, min_neighbor_dist)
    r_raw = np.maximum(r_raw, 0.0)
    
    # Penalty for violations: if any constraint is violated (negative radius component),
    # add a quadratic penalty. Here we use a smooth approximation.
    # Actually, let's use a different approach: 
    # Objective = sum(r_i) - mu * sum(max(0, violation_i)^2)
    # But we need the gradient. Let's use a smooth max.
    
    # Better approach: use log-sum-exp style or just penalize negative parts
    # Let's define: for each circle, the limiting constraint
    # If dist_bound < min_neighbor_dist, the boundary limits
    # Otherwise, the nearest neighbor limits
    
    # Gradient computation:
    # d/dx_i of r_i where r_i = min(dist_bound_i, min_j dists[i,j]/2)
    
    # Let's compute which constraint is active for each circle
    active_is_boundary = dist_bound < min_neighbor_dist
    
    # Gradient of dist_bound w.r.t. x_i
    # dist_bound_i = min(x_i0, 1-x_i0, x_i1, 1-x_i1)
    grad_dist_bound = np.zeros((N, 2))
    for i in range(N):
        d0 = x[i, 0]
        d1 = 1 - x[i, 0]
        d2 = x[i, 1]
        d3 = 1 - x[i, 1]
        min_d = min(d0, d1, d2, d3)
        if abs(d0 - min_d) < 1e-10:
            grad_dist_bound[i, 0] += 1.0
        if abs(d1 - min_d) < 1e-10:
            grad_dist_bound[i, 0] += -1.0
        if abs(d2 - min_d) < 1e-10:
            grad_dist_bound[i, 1] += 1.0
        if abs(d3 - min_d) < 1e-10:
            grad_dist_bound[i, 1] += -1.0
    
    # Gradient of min_neighbor_dist w.r.t. x_i
    # min_neighbor_dist_i = min_j dists[i,j] / 2
    # d/dx_i of dists[i,j] = (x_i - x_j) / dists[i,j]
    grad_min_neighbor = np.zeros((N, 2))
    for i in range(N):
        # Find the nearest neighbor (excluding self)
        dists_i = dists[i].copy()
        dists_i[i] = np.inf
        j_nearest = np.argmin(dists_i)
        if dists_i[j_nearest] > 1e-10:
            grad_min_neighbor[i] = (x[i] - x[j_nearest]) / (2.0 * dists_i[j_nearest])
    
    # Gradient of r_i
    grad_r = np.zeros((N, 2))
    for i in range(N):
        if active_is_boundary[i]:
            grad_r[i] = grad_dist_bound[i]
        else:
            grad_r[i] = grad_min_neighbor[i]
    
    # Total objective: -sum(r_i)
    val = -np.sum(r_raw)
    grad = -grad_r.ravel()
    
    # Add penalty for circles that are too close to boundary or each other
    # This helps L-BFGS-B stay in feasible region
    penalty = 0.0
    grad_penalty = np.zeros_like(flat_x)
    for i in range(N):
        # Penalty for being outside [0,1]
        if x[i, 0] < 0.01:
            penalty += mu * (0.01 - x[i, 0])**2
            grad_penalty[2*i] += 2*mu*(x[i, 0] - 0.01)
        elif x[i, 0] > 0.99:
            penalty += mu * (x[i, 0] - 0.99)**2
            grad_penalty[2*i] += 2*mu*(x[i, 0] - 0.99)
        if x[i, 1] < 0.01:
            penalty += mu * (0.01 - x[i, 1])**2
            grad_penalty[2*i+1] += 2*mu*(x[i, 1] - 0.01)
        elif x[i, 1] > 0.99:
            penalty += mu * (x[i, 1] - 0.99)**2
            grad_penalty[2*i+1] += 2*mu*(x[i, 1] - 0.99)
    
    val += penalty
    grad += grad_penalty
    
    return val, grad

def _hex_init(n, rng):
    """Generate hexagonal packing initialization."""
    points = []
    for q in range(-10, 11):
        for r in range(-10, 11):
            x = 1.5 * q
            y = np.sqrt(3) * (r + 0.5 * (q & 1))
            points.append([x, y])
    points = np.array(points)
    
    # Center
    center = np.array([0.5, 0.5])
    points += (center - np.mean(points, axis=0))
    
    # Select n closest to center
    dists = np.sum((points - center)**2, axis=1)
    sorted_idx = np.argsort(dists)
    init_points = points[sorted_idx[:n]]
    
    return init_points

def _random_init(n, rng):
    """Generate random initialization with minimum spacing."""
    points = np.zeros((n, 2))
    # Start with a grid and add noise
    grid_size = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        gx = i % grid_size
        gy = i // grid_size
        points[i, 0] = (gx + 0.5) / grid_size + rng.uniform(-0.05, 0.05)
        points[i, 1] = (gy + 0.5) / grid_size + rng.uniform(-0.05, 0.05)
    
    # Ensure inside [0.05, 0.95]
    points = np.clip(points, 0.05, 0.95)
    
    return points

def _is_valid(flat_x, tol=1e-6):
    """Check if configuration is valid (all circles inside and non-overlapping)."""
    x = flat_x.reshape(N, 2)
    
    # Check boundaries
    if np.any(x < -tol) or np.any(x > 1 + tol):
        return False
    
    # Check pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Compute radii
    dist_bound = np.minimum(np.minimum(x[:, 0], 1 - x[:, 0]), np.minimum(x[:, 1], 1 - x[:, 1]))
    min_dist_other = np.min(dists, axis=1) / 2.0
    radii = np.minimum(dist_bound, min_dist_other)
    
    # All radii should be non-negative
    if np.any(radii < -tol):
        return False
    
    return True

def _optimize(x0, deadline):
    """Run L-BFGS-B optimization from initial point."""
    best_x = x0.copy()
    best_val, _ = _compute_penalty_and_grad(x0)
    
    # First pass: L-BFGS-B
    try:
        result = minimize(
            _compute_penalty_and_grad, 
            x0, 
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    # Second pass: Powell for local refinement
    if time.time() < deadline - 5:
        try:
            def obj_only(flat_x):
                return _compute_penalty_and_grad(flat_x)[0]
            
            result2 = minimize(
                obj_only,
                best_x,
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-6, 'ftol': 1e-8}
            )
            if result2.fun < best_val:
                best_val = result2.fun
                best_x = result2.x
        except Exception:
            pass
    
    return best_x, best_val

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_score = -np.inf
    
    # Generate multiple diverse initializations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15:
            break
        
        # Alternate between hex and random inits
        if restart % 2 == 0:
            init_points = _hex_init(N, rng)
        else:
            init_points = _random_init(N, rng)
        
        x0 = init_points.ravel()
        
        # Ensure initial point is valid
        if not _is_valid(x0):
            # Try to fix by clipping
            x0 = np.clip(x0, 0.02, 0.98)
        
        # Optimize
        result_x, result_val = _optimize(x0, deadline)
        
        # Compute actual score (sum of radii)
        final_centers = np.clip(result_x.reshape(N, 2), 0, 1)
        dist_bound = np.minimum(np.minimum(final_centers[:, 0], 1 - final_centers[:, 0]), 
                                np.minimum(final_centers[:, 1], 1 - final_centers[:, 1]))
        diff = final_centers[:, np.newaxis, :] - final_centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist_other = np.min(dists, axis=1) / 2.0
        radii = np.minimum(dist_bound, min_dist_other)
        radii = np.maximum(radii, 0)
        
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_x = result_x
    
    # If no valid result found, use a fallback
    if best_x is None:
        init_points = _hex_init(N, rng)
        best_x = init_points.ravel()
    
    # Final calculation
    final_centers = np.clip(best_x.reshape(N, 2), 0, 1)
    dist_bound = np.minimum(np.minimum(final_centers[:, 0], 1 - final_centers[:, 0]), 
                            np.minimum(final_centers[:, 1], 1 - final_centers[:, 1]))
    diff = final_centers[:, np.newaxis, :] - final_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_other = np.min(dists, axis=1) / 2.0
    radii = np.minimum(dist_bound, min_dist_other)
    radii = np.maximum(radii, 0)
    
    return np.column_stack([final_centers, radii])
