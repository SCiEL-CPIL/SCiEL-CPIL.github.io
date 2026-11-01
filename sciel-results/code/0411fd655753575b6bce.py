import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12
GRID_SIZE = 20000

def _generate_grid(M):
    """Generate a Fibonacci lattice grid."""
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    return _generate_grid(N)

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_hull_radius(pts):
    """
    Compute the true covering radius using the convex hull.
    This is the ground truth for scoring.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Compute normals for each face
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        # Filter degenerate
        valid = norms.flatten() > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid] / (norms[valid] + _EPS)
        
        # Ensure outward direction
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        
        # The pole is the normalized normal.
        # Distance is arccos(pole . vertex)
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        return np.max(radii)
    except Exception:
        return np.pi

def _soft_hull_objective_and_grad(pts, grid, temperature):
    """
    Compute a differentiable surrogate for the covering radius.
    
    For each grid point g, let d_g = min_i (1 - g . p_i).
    The covering radius is max_g d_g.
    We approximate max_g d_g with a log-sum-exp of d_g.
    
    Actually, it's better to work with angular distances directly or cosines.
    Let theta_g = arccos(max_i g . p_i).
    We want to minimize max_g theta_g.
    
    Using a log-sum-exp on the angles is smooth.
    However, arccos is not differentiable at 1.
    Let's use the squared distances or just the cosines?
    Minimizing max theta is equivalent to minimizing max (1 - cos theta) for small theta? No.
    
    Let's stick to the soft-max of the angular distances.
    L = (1/T) * log( sum_g exp(T * theta_g) )
    
    Gradient of L w.r.t p_i:
    dL/dp_i = sum_g (w_g * d(theta_g)/d(p_i))
    where w_g = exp(T * theta_g) / sum_k exp(T * theta_k)
    
    d(theta_g)/d(p_i) = - (g - (g.p_i) p_i) / sqrt(1 - (g.p_i)^2)  (if p_i is the nearest)
    But since theta_g = arccos(max_j g.p_j), the derivative is non-zero only for the argmax j.
    This makes the gradient sparse but discontinuous.
    
    A smoother approach:
    Use the "soft" nearest neighbor.
    Let alpha_j = exp(T * g.p_j) / sum_k exp(T * g.p_k).
    Then soft_cos_theta_g = sum_j alpha_j g.p_j.
    soft_theta_g = arccos(soft_cos_theta_g).
    
    Objective: L = (1/T) log( sum_g exp(T * soft_theta_g) )
    
    This is fully differentiable.
    """
    # pts: (N, 3), grid: (M, 3)
    # dots: (M, N) = grid @ pts.T
    dots = grid @ pts.T
    
    # Softmax over N for each grid point
    # Subtract max for stability
    max_dots = np.max(dots, axis=1, keepdims=True)
    exp_dots = np.exp(T * (dots - max_dots))
    sum_exp = np.sum(exp_dots, axis=1, keepdims=True)
    alpha = exp_dots / (sum_exp + _EPS)  # (M, N)
    
    # soft_cos_theta = sum_j alpha_j * dots[:, j]
    soft_cos_theta = np.sum(alpha * dots, axis=1)  # (M,)
    soft_cos_theta = np.clip(soft_cos_theta, -1.0 + _EPS, 1.0 - _EPS)
    
    soft_theta = np.arccos(soft_cos_theta)  # (M,)
    
    # Log-sum-exp of soft_theta
    max_theta = np.max(soft_theta)
    exp_theta = np.exp(T * (soft_theta - max_theta))
    sum_exp_theta = np.sum(exp_theta)
    L = (1.0 / T) * (max_theta + np.log(sum_exp_theta + _EPS))
    
    # Gradient
    # dL/dp_i = (1/T) * (1/sum_exp_theta) * sum_g exp(T * theta_g) * d(theta_g)/dp_i
    # d(theta_g)/dp_i = -1/sqrt(1-cos^2) * d(cos_theta_g)/dp_i
    # d(cos_theta_g)/dp_i = sum_j alpha_j * g - (sum_j alpha_j * (g.p_j)) * (g.p_i * p_i - (g.p_i) g)?
    # Wait. cos_theta_g = sum_j alpha_j (g . p_j).
    # d/dp_i (g . p_j) = g if i==j else 0.
    # But alpha_j also depends on p_i.
    # This gets complicated.
    
    # Alternative: Use the subgradient of the max function directly on the grid.
    # For each grid point, find the nearest point index k_g.
    # If we treat the max as a hard max, the gradient is 0 for all points except the nearest.
    # For the nearest point k_g, the gradient of theta_g w.r.t p_{k_g} is the projection of -g onto the tangent space at p_{k_g}.
    # grad_theta = - (g - (g.p_k) p_k) / ||g - (g.P_k) p_k|| ?
    # Actually, theta = arccos(g.p). dtheta/dp = -g / sqrt(1-(g.p)^2) projected?
    # Let's use the vectorized hard-max gradient with a slight smoothing if needed, 
    # or just use the subgradient which is often effective.
    
    # Let's try the hard-max subgradient first, it's simpler and faster.
    # nearest_idx = argmax(dots, axis=1)
    # For each grid point g, let k = nearest_idx[g].
    # grad_theta_g = - (g - (g.p_k) p_k) / sin(theta_g)
    # The total gradient for p_i is the sum of grad_theta_g for all g where k==i,
    # weighted by the softmax weights w_g = exp(T * theta_g) / sum exp.
    
    # Let's compute theta_g using the hard max for the gradient calculation to save time,
    # but use the soft L for the value? No, let's be consistent.
    
    # Let's use the hard max for gradient direction, it's a common trick.
    nearest_idx = np.argmax(dots, axis=1)
    hard_cos_theta = np.max(dots, axis=1)
    hard_theta = np.arccos(np.clip(hard_cos_theta, -1.0 + _EPS, 1.0 - _EPS))
    
    # Weights for the log-sum-exp
    max_hard_theta = np.max(hard_theta)
    exp_hard_theta = np.exp(T * (hard_theta - max_hard_theta))
    sum_exp_hard = np.sum(exp_hard_theta)
    weights = exp_hard_theta / (sum_exp_hard + _EPS)
    
    # Gradient of theta_g w.r.t p_k
    # theta_g = arccos(g.p_k)
    # dtheta/dp_k = -g / sqrt(1 - (g.p_k)^2)
    # But we need to project onto the tangent space? 
    # Riemannian gradient is the projection of the Euclidean gradient onto the tangent space.
    # Euclidean grad: -g / sin(theta_g)
    # Tangent projection: (I - p_k p_k^T) * ( -g / sin(theta_g) )
    # = - (g - (g.p_k) p_k) / sin(theta_g)
    
    sin_theta = np.sin(hard_theta)
    sin_theta = np.clip(sin_theta, _EPS, None)
    
    # For each grid point g, we have a vector v_g = - (g - (g.p_k) p_k) / sin(theta_g)
    # We need to sum these vectors for each point p_i, weighted by weights_g.
    
    # v_g is a vector in R^3.
    # p_k is pts[nearest_idx[g]]
    # g is grid[g]
    # g.p_k is hard_cos_theta[g]
    
    # v_g = - (grid[g] - hard_cos_theta[g] * pts[nearest_idx[g]]) / sin_theta[g]
    
    # We can compute this vectorized:
    # nearest_pts = pts[nearest_idx]  # (M, 3)
    # v = - (grid - hard_cos_theta[:, None] * nearest_pts) / sin_theta[:, None]
    
    nearest_pts = pts[nearest_idx]
    v = - (grid - hard_cos_theta[:, None] * nearest_pts) / sin_theta[:, None]
    
    # Now we need to accumulate v[g] * weights[g] into grad[nearest_idx[g]]
    grad = np.zeros((N, 3))
    for k in range(3):
        grad[:, k] = np.bincount(nearest_idx, weights=weights * v[:, k], minlength=N)
        
    # Riemannian gradient is already projected?
    # The formula - (g - (g.p) p) / sin(theta) is already tangent to the sphere at p.
    # So this is the Riemannian gradient.
    
    # Scale by 1/T? 
    # The objective was L = (1/T) log sum exp(T theta).
    # The gradient of L is (1/T) * sum w_g T grad_theta_g = sum w_g grad_theta_g.
    # So the 1/T and T cancel out. The gradient is just the weighted sum of subgradients.
    
    return L, grad

def _riemannian_grad_descent(pts, grid, n_iters, temperature=10.0, lr=0.01):
    """
    Perform Riemannian gradient descent on the sphere.
    """
    for _ in range(n_iters):
        _, grad = _soft_hull_objective_and_grad(pts, grid, temperature)
        
        # Update: p_new = p - lr * grad
        pts = pts - lr * grad
        
        # Retract onto the sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / (norms + _EPS)
        
    return pts

def _lloyd_relax(pts, grid, n_iters=10):
    """
    Standard Lloyd relaxation for comparison/fallback.
    """
    pts = pts.copy()
    M = grid.shape[0]
    
    for _ in range(n_iters):
        dots = grid @ pts.T
        nearest_idx = np.argmax(dots, axis=1)
        
        new_pts = np.zeros((N, 3))
        for k in range(3):
            new_pts[:, k] = np.bincount(nearest_idx, weights=grid[:, k], minlength=N)
            
        counts = np.bincount(nearest_idx, minlength=N).astype(float)
        valid = counts > 0
        norms = np.linalg.norm(new_pts[valid], axis=1, keepdims=True)
        new_pts[valid] /= (norms + _EPS)
        new_pts[~valid] = pts[~valid]
        
        pts = new_pts
        
    return pts

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    grid = _generate_grid(GRID_SIZE)
    
    best_pts = _initial_fib()
    best_r = _compute_hull_radius(best_pts)
    
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = _random_init(rng)
            
        # Phase 1: Coarse Gradient Descent
        # Use a higher temperature for smoother landscape
        phase_deadline = time.time() + 10.0
        iters = 0
        while time.time() < phase_deadline and iters < 500:
            pts = _riemannian_grad_descent(pts, grid, n_iters=10, temperature=20.0, lr=0.05)
            iters += 10
            
        # Phase 2: Fine Gradient Descent
        phase_deadline = time.time() + 15.0
        iters = 0
        while time.time() < phase_deadline and iters < 500:
            pts = _riemannian_grad_descent(pts, grid, n_iters=10, temperature=5.0, lr=0.01)
            iters += 10
            
        # Phase 3: Lloyd Polish
        phase_deadline = time.time() + 5.0
        while time.time() < phase_deadline:
            pts = _lloyd_relax(pts, grid, n_iters=5)
            
        curr_r = _compute_hull_radius(pts)
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
            
    # Final polish on the best configuration
    if time.time() < deadline - 5.0:
        phase_deadline = time.time() + 10.0
        while time.time() < phase_deadline:
            best_pts = _lloyd_relax(best_pts, grid, n_iters=10)
            curr_r = _compute_hull_radius(best_pts)
            if curr_r < best_r:
                best_r = curr_r
            else:
                break
                
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
