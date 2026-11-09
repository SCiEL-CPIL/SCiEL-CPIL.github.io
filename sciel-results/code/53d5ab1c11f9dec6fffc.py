"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute covering radius in radians via convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            pts = points[simplex]
            # Centroid of the facet
            c = np.mean(pts, axis=0)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-10:
                continue
            c = c / c_norm
            n = np.cross(pts[0] - pts[1], pts[0] - pts[2])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            x = n / n_norm
            if np.dot(x, c) < 0:
                x = -x
            cos_dist = np.dot(x, pts[0])
            cos_dist = np.clip(cos_dist, -1.0, 1.0)
            ang = np.arccos(cos_dist)
            max_r = max(max_r, ang)
        return max_r
    except Exception:
        return np.pi

def _soft_max_surrogate(points, beta=50.0):
    """
    Fast O(N^2) surrogate for covering radius optimization.
    Maximizes the minimum pairwise angular separation using a soft-max penalty.
    Returns (objective_value, gradient) where objective is to be minimized.
    The objective is -min_i (max_j dot(x_i, x_j)) approximated via soft-max.
    """
    n = len(points)
    # Compute all pairwise dot products
    dots = points @ points.T
    np.fill_diagonal(dots, -1.0)  # Exclude self
    
    # For each point i, find the maximum dot product with any other point j
    # This corresponds to the minimum angular distance (closest neighbor)
    # We want to MAXIMIZE this minimum, so we MINIMIZE the negative
    
    # Soft-max approximation of max_j dots[i,j]
    # Using log-sum-exp for numerical stability
    max_dots = np.max(dots, axis=1)
    # Shift for stability
    shifted = dots - max_dots[:, np.newaxis]
    exp_shifted = np.exp(beta * shifted)
    # Handle potential overflow by clipping
    exp_shifted = np.clip(exp_shifted, 0, 1e100)
    log_sum_exp = np.log(np.sum(exp_shifted, axis=1)) + max_dots / beta
    
    # The objective is to minimize the negative of the soft-max of max dots
    # i.e., minimize -soft_max, which means maximize the closest-neighbor dot product
    objective = -np.mean(log_sum_exp)
    
    # Gradient computation
    # d/dx_i of log_sum_exp[i] where log_sum_exp[i] = log(sum_j exp(beta*(dots[i,j] - max_dots[i]))) + max_dots[i]/beta
    # For simplicity, use the full soft-max gradient
    # d log_sum_exp[i] / d dots[i,j] = beta * exp(beta*(dots[i,j]-max_dots[i])) / sum_k exp(beta*(dots[i,k]-max_dots[k]))
    
    weights = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    weights = np.clip(weights, 0, 1)
    # Ensure row sums are 1
    row_sums = np.sum(weights, axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-15, 1.0, row_sums)
    weights = weights / row_sums
    
    # Gradient of dots[i,j] w.r.t. x_i is x_j, w.r.t. x_j is x_i
    # d objective / d x_i = -beta * sum_j weights[i,j] * x_j (from term i)
    #                      -beta * sum_j weights[j,i] * x_i (from term j, since dots[j,i] = dots[i,j])
    
    # First term: for each i, sum over j of weights[i,j] * x_j
    term1 = weights @ points  # shape (n, 3)
    
    # Second term: for each i, sum over j of weights[j,i] * x_i = (sum_j weights[j,i]) * x_i
    col_sums = np.sum(weights, axis=0)  # shape (n,)
    term2 = col_sums[:, np.newaxis] * points  # shape (n, 3)
    
    gradient = -beta * (term1 + term2) / n
    
    return objective, gradient

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _fibonacci_sphere(n, rng=None):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_soft_max(points, max_iters=500, lr_init=0.05, beta=50.0, tol=1e-10):
    """
    Projected gradient ascent to maximize the minimum pairwise dot product
    using a soft-max surrogate.
    """
    pts = points.copy()
    best_pts = pts.copy()
    best_obj = _soft_max_surrogate(pts, beta)[0]
    
    lr = lr_init
    momentum = np.zeros_like(pts)
    mu = 0.9  # Momentum coefficient
    
    for it in range(max_iters):
        obj, grad = _soft_max_surrogate(pts, beta)
        
        # Update momentum
        momentum = mu * momentum - lr * grad
        
        # Take step
        new_pts = pts + momentum
        
        # Project to sphere
        new_pts = _project_to_sphere(new_pts)
        
        # Check if objective improved
        new_obj = _soft_max_surrogate(new_pts, beta)[0]
        
        if new_obj < best_obj:
            best_obj = new_obj
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Line search: reduce step size
            lr *= 0.8
            # Revert
            momentum = mu * momentum
    
        # Decay learning rate
        if it % 20 == 0 and lr > 0.001:
            lr *= 0.95
        
        # Convergence check
        if it > 50 and abs(obj - best_obj) < tol:
            break
    
    return best_pts, best_obj

def _optimize_combined(points, max_iters=200, tol=1e-8):
    """
    Combined optimization: first use soft-max surrogate for fast convergence,
    then polish with convex hull-based covering radius.
    """
    pts = points.copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Phase 1: Soft-max optimization
    soft_pts, _ = _optimize_soft_max(pts, max_iters=300, beta=50.0)
    cr1 = _covering_radius(soft_pts)
    if cr1 < best_cr:
        best_cr = cr1
        best_pts = soft_pts.copy()
    
    # Phase 2: Fine-tune with gradient-based approach on covering radius
    # Use the soft-max gradient as a proxy for covering radius gradient
    lr = 0.01
    for it in range(max_iters):
        obj, grad = _soft_max_surrogate(best_pts, beta=100.0)
        
        # Project gradient to tangent space
        tangent_grad = grad - np.sum(grad * best_pts, axis=1, keepdims=True) * best_pts
        
        new_pts = best_pts + lr * tangent_grad
        new_pts = _project_to_sphere(new_pts)
        
        cr = _covering_radius(new_pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = new_pts
        else:
            lr *= 0.9
        
        if lr < 1e-5:
            break
    
    return best_pts, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_cr = np.pi
    
    rng = np.random.default_rng(42)
    
    # Multiple initializations
    initializations = []
    
    # Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    initializations.append(pts_fib)
    
    # Random
    pts_rand = rng.standard_normal((N, 3))
    pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
    initializations.append(pts_rand)
    
    # Perturbed Fibonacci
    pts_pert = pts_fib + 0.1 * rng.standard_normal((N, 3))
    pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
    initializations.append(pts_pert)
    
    # Different golden angle
    i4 = np.arange(N) + 0.5
    z4 = 1.0 - 2.0 * i4 / N
    phi4 = np.pi * (1.0 + np.sqrt(5.0)) * i4 * 1.618
    r4 = np.sqrt(np.maximum(0, 1.0 - z4 * z4))
    pts_ga = np.column_stack([r4 * np.cos(phi4), r4 * np.sin(phi4), z4])
    initializations.append(pts_ga)
    
    # More random restarts
    for _ in range(3):
        pts_r = rng.standard_normal((N, 3))
        pts_r /= np.linalg.norm(pts_r, axis=1, keepdims=True)
        initializations.append(pts_r)
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Run combined optimization
        opt_pts, cr = _optimize_combined(init_pts, max_iters=150)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = opt_pts.copy()
        
        # If we have time, do additional fine-tuning on the best so far
        if time.time() < deadline - 20 and best_pts is not None:
            # Fine-tune the current best
            fine_pts, fine_cr = _optimize_combined(best_pts, max_iters=100)
            if fine_cr < best_cr:
                best_cr = fine_cr
                best_pts = fine_pts.copy()
    
    if best_pts is None:
        best_pts = initializations[0]
    
    return best_pts
# EVOLVE-BLOCK-END
