"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1)
    norms[norms < 1e-10] = 1.0
    return pts / norms[:, None]

def _covering_radius_deg(pts):
    """Compute exact covering radius in degrees. Returns inf if invalid."""
    pts = np.asarray(pts)
    if pts.shape != (N, 3):
        return np.inf
    if not np.all(np.isfinite(pts)):
        return np.inf
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-10):
        return np.inf
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.inf
    
    max_rad = 0.0
    for simplex in hull.simplices:
        v = pts[simplex]
        k = len(simplex)
        G = v @ v.T
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            continue
        s = np.ones(k)
        val = s @ G_inv @ s
        if val <= 0:
            continue
        cos_theta = 1.0 / np.sqrt(val)
        if cos_theta >= 1.0:
            continue
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        rad_deg = np.degrees(theta)
        if rad_deg > max_rad:
            max_rad = rad_deg
    return max_rad

def _smooth_covering_objective(x_flat):
    """
    Smooth surrogate objective for optimization.
    Uses a dense grid of test points on the sphere.
    Objective: max over grid points of (distance to nearest config point).
    To make it differentiable, we use a soft-max approximation or simply 
    optimize the sum of squared distances to the 'worst' grid points.
    However, a simpler robust approach for L-BFGS is to minimize the 
    maximum distance directly via a smooth approximation or just use 
    the grid to define a penalty.
    
    Actually, L-BFGS works well on smooth functions. The max function is not smooth.
    We can use the 'soft-max' trick: 
    f(x) = (1/beta) * log(sum(exp(beta * d_i^2)))
    where d_i is distance from grid point i to nearest config point.
    This approximates max(d_i^2) as beta -> inf.
    """
    pts = _normalize(x_flat.reshape(N, 3))
    
    # Pre-computed grid of test points (created outside for speed, but here for self-containment 
    # in the function signature if we don't pass it, we use a global or recreate. 
    # Better: define grid in construct and pass via closure or global. 
    # For simplicity in this block, we use a moderately dense fixed grid.
    # To be efficient, we should precompute the grid in construct. 
    # But since we can't easily pass args to minimize's fun without closure, 
    # we'll define the grid once in construct and use a closure.
    
    # Wait, the function signature is fixed. I will define a helper that takes the grid.
    # I'll implement the logic in construct using a nested function.
    
    # Placeholder for standalone check if called directly (unlikely in this flow)
    return 0.0 

def _create_fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 10s safety margin
    
    # Pre-compute a dense grid of test points on the sphere for the smooth objective.
    # We use a Fibonacci sphere with ~5000 points to approximate the continuous max.
    M = 5000
    grid_pts = _create_fibonacci_sphere(M)
    
    # Helper to compute distances from grid points to config points
    # Returns array of shape (M,) with min distance for each grid point
    
    def _min_distances(pts):
        # pts: (N, 3), grid_pts: (M, 3)
        # Compute squared distances: ||g - p||^2 = ||g||^2 + ||p||^2 - 2 g.p = 2 - 2 g.p
        # Since all are unit vectors.
        # dot products: (M, N)
        dots = grid_pts @ pts.T  # (M, N)
        min_dots = np.max(dots, axis=1)  # (M,) max dot product corresponds to min distance
        # cos(theta) = min_dots
        # theta = arccos(min_dots)
        # We want to minimize theta.
        # Objective: sum of soft-max of theta? Or just max?
        # L-BFGS prefers smooth. 
        # Let's use: (1/beta) * log(sum(exp(beta * theta_i)))
        # theta_i = arccos(min_dots_i)
        # Note: arccos is smooth for inputs in (-1, 1). 
        # We clip min_dots to [-1+eps, 1-eps] to avoid singularities.
        min_dots = np.clip(min_dots, -1 + 1e-9, 1 - 1e-9)
        thetas = np.arccos(min_dots)
        
        # Soft-max approximation of max(theta)
        # beta controls the sharpness. Higher beta -> closer to max.
        # We start with moderate beta and can increase if time permits, 
        # but L-BFGS with high beta can be ill-conditioned.
        # Let's try beta = 50.0 initially.
        beta = 50.0
        # For numerical stability: max_val = max(thetas)
        # soft_max = max_val + (1/beta) * log(sum(exp(beta * (thetas - max_val))))
        max_theta = np.max(thetas)
        exp_terms = np.exp(beta * (thetas - max_theta))
        log_sum_exp = np.log(np.sum(exp_terms))
        soft_max_theta = max_theta + (1.0 / beta) * log_sum_exp
        
        # Return in radians. We will convert to degrees at the end.
        # Minimizing soft_max_theta is equivalent to minimizing covering radius.
        return soft_max_theta

    def _objective_and_grad(x_flat):
        pts = _normalize(x_flat.reshape(N, 3))
        
        # Compute min dots
        dots = grid_pts @ pts.T
        min_dots = np.max(dots, axis=1)
        min_dots = np.clip(min_dots, -1 + 1e-9, 1 - 1e-9)
        thetas = np.arccos(min_dots)
        
        beta = 50.0
        max_theta = np.max(thetas)
        scaled = beta * (thetas - max_theta)
        # Clip scaled to avoid overflow in exp
        scaled = np.clip(scaled, -50, 50)
        exp_terms = np.exp(scaled)
        sum_exp = np.sum(exp_terms)
        
        # Objective value
        obj_val = max_theta + (1.0 / beta) * np.log(sum_exp)
        
        # Gradient calculation
        # d obj / d theta_i = (1/beta) * exp(scaled_i) / sum_exp = (1/beta) * p_i
        # where p_i = exp(scaled_i) / sum_exp (probability distribution over grid points)
        p = exp_terms / sum_exp
        
        # d theta_i / d min_dots_i = -1 / sqrt(1 - min_dots_i^2)
        # Note: min_dots_i = max_j (grid_i . pts_j)
        # Let j_i be the index of the point in pts that is closest to grid_i.
        # d min_dots_i / d pts_j = grid_i if j == j_i, 0 otherwise.
        
        # So d obj / d pts_j = sum_{i: j_i == j} p_i * (-1/sqrt(1-min_dots_i^2)) * grid_i
        
        # Find j_i for each i
        j_indices = np.argmax(dots, axis=1)
        
        # Gradient accumulation
        grad_pts = np.zeros_like(pts)
        # We need 1/sqrt(1 - min_dots^2)
        # denom = sqrt(1 - min_dots^2)
        denom = np.sqrt(1.0 - min_dots * min_dots)
        # Clip to avoid division by zero
        denom = np.clip(denom, 1e-9, None)
        
        # Factor for each grid point
        factors = -p / denom
        
        # Accumulate gradients
        # For each grid point i, add factors[i] * grid_pts[i] to grad_pts[j_indices[i]]
        # Using np.add.at for efficient accumulation
        np.add.at(grad_pts, j_indices, factors[:, None] * grid_pts)
        
        # The gradient w.r.t x_flat is the same as w.r.t pts, 
        # but we must account for the normalization projection?
        # Actually, x_flat are the raw coordinates. 
        # If we treat the optimization as happening on the sphere, 
        # we should project the gradient onto the tangent space.
        # grad_tangent = grad_pts - pts * (pts . grad_pts)
        
        proj_factors = np.sum(pts * grad_pts, axis=1, keepdims=True)
        grad_tangent = grad_pts - pts * proj_factors
        
        # Reshape to flat
        grad_flat = grad_tangent.flatten()
        
        return obj_val, grad_flat

    best_pts = None
    best_cr = np.inf
    
    # Multi-start strategy
    num_starts = 5
    rng = np.random.default_rng(42)
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 20:
            break
            
        # Initialize
        if start_idx == 0:
            init_pts = _create_fibonacci_sphere(N)
        else:
            # Perturbed random or Fibonacci
            init_pts = _create_fibonacci_sphere(N)
            noise = rng.normal(0, 0.1, (N, 3))
            init_pts = _normalize(init_pts + noise)
        
        x0 = init_pts.flatten()
        
        # L-BFGS-B optimization
        try:
            res = minimize(
                _objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                options={
                    'maxiter': 200,
                    'ftol': 1e-9,
                    'gtol': 1e-6,
                    'maxls': 20
                }
            )
        except Exception:
            continue
        
        opt_pts = _normalize(res.x.reshape(N, 3))
        
        # Evaluate exact covering radius
        cr = _covering_radius_deg(opt_pts)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = opt_pts.copy()
            
        # If we found a very good solution, we can stop or continue
        # 11.31 is the target. If we are below 11.5, we are doing well.
        if best_cr < 11.4:
            # Try one more polish with higher beta? 
            # Or just break to save time for other starts?
            # Given time constraints, break if we are very close to known best.
            if best_cr < 11.32:
                break

    # Final fallback if optimization failed completely
    if best_pts is None:
        best_pts = _create_fibonacci_sphere(N)
        best_cr = _covering_radius_deg(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
