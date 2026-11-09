"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius(points):
    """Compute covering radius in radians via convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            pts = points[simplex]
            # Find circumcenter of spherical triangle
            # The point on the sphere farthest from all three vertices
            # For a spherical triangle, the circumcenter direction is:
            a, b, c = pts[0], pts[1], pts[2]
            # Solve for circumcenter: point x on sphere equidistant from a, b, c
            # x · a = x · b = x · c (in terms of cos of angular distance)
            # This means x is perpendicular to (a-b) and (a-c)
            ab = a - b
            ac = a - c
            n = np.cross(ab, ac)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            x = n / n_norm
            # Check if x is on the correct side (same side as centroid)
            centroid = (a + b + c) / 3.0
            if np.dot(x, centroid) < 0:
                x = -x
            # Compute angular distance from x to each vertex
            cos_a = np.clip(np.dot(x, a), -1.0, 1.0)
            cos_b = np.clip(np.dot(x, b), -1.0, 1.0)
            cos_c = np.clip(np.dot(x, c), -1.0, 1.0)
            ang = max(np.arccos(cos_a), np.arccos(cos_b), np.arccos(cos_c))
            max_r = max(max_r, ang)
        return max_r
    except Exception:
        return np.pi

def _nearest_neighbor_dists(points):
    """For each point, find the nearest neighbor distance (angular)."""
    dots = points @ points.T
    np.fill_diagonal(dots, -2.0)  # Ensure self is not chosen
    max_dots = np.max(dots, axis=1)
    max_dots = np.clip(max_dots, -1.0, 1.0)
    return np.arccos(max_dots)

def _soft_max_min_separation(points, beta=30.0):
    """
    Soft-max surrogate: minimize -mean(soft_max of min separation per point).
    For each point i, the closest neighbor defines the local density.
    We want to maximize the minimum over i of (closest neighbor angular distance).
    """
    n = len(points)
    dots = points @ points.T
    np.fill_diagonal(dots, -2.0)
    
    # For each i, we want max_j dots[i,j] (closest neighbor in dot space = min angular dist)
    # soft-max of max: log-sum-exp
    max_dots = np.max(dots, axis=1, keepdims=True)
    shifted = dots - max_dots
    exp_shifted = np.exp(beta * shifted)
    exp_shifted = np.clip(exp_shifted, 1e-300, 1e300)
    log_sum = np.log(np.sum(exp_shifted, axis=1, keepdims=True))
    soft_max = (log_sum + max_dots) / beta
    
    # Objective: minimize -mean(soft_max) = maximize mean(soft_max)
    objective = -np.mean(soft_max)
    
    # Gradient
    weights = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    weights = np.clip(weights, 0, 1)
    row_sums = np.sum(weights, axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-15, 1.0, row_sums)
    weights = weights / row_sums
    
    # d objective / d x_i = -(1/n) * [beta * sum_j weights[i,j] * x_j + beta * (sum_j weights[j,i]) * x_i]
    term1 = weights @ points
    col_sums = np.sum(weights, axis=0)
    term2 = col_sums[:, np.newaxis] * points
    gradient = -beta * (term1 + term2) / n
    
    return objective, gradient

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms

def _fibonacci_sphere(n):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _repulsion_force(points, k=10):
    """
    Compute repulsive force: each point is pushed away from its k nearest neighbors.
    This creates uniform spacing.
    """
    n = len(points)
    dots = points @ points.T
    np.fill_diagonal(dots, -2.0)
    
    # Find k nearest neighbors for each point
    # Use argsort to get indices of k largest dot products (closest neighbors)
    nn_indices = np.argsort(-dots, axis=1)[:, :k]
    
    forces = np.zeros_like(points)
    for i in range(n):
        nn_idx = nn_indices[i]
        # Force from each neighbor: push away proportional to closeness
        for j in nn_idx:
            d = dots[i, j]
            if d > 0.5:  # Only repel if quite close
                # Gradient of dot product w.r.t. x_i is x_j
                forces[i] -= d * points[j]
    
    return forces

def _optimize_repulsion(points, max_iters=200, lr=0.01, k=8):
    """Optimize using repulsion forces from k-nearest neighbors."""
    pts = points.copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    for it in range(max_iters):
        forces = _repulsion_force(pts, k=k)
        
        # Project to tangent space
        tangent_forces = forces - np.sum(forces * pts, axis=1, keepdims=True) * pts
        
        new_pts = pts + lr * tangent_forces
        new_pts = _project_to_sphere(new_pts)
        
        cr = _covering_radius(new_pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            lr *= 0.95
        
        if lr < 1e-6:
            break
    
    return best_pts, best_cr

def _optimize_softmax_polish(points, max_iters=150, lr=0.02, beta=30.0):
    """Polish using soft-max gradient with tangent projection."""
    pts = points.copy()
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    momentum = np.zeros_like(pts)
    mu = 0.5
    
    for it in range(max_iters):
        obj, grad = _soft_max_min_separation(pts, beta=beta)
        
        # Project gradient to tangent space
        tangent_grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Momentum update
        momentum = mu * momentum + lr * tangent_grad
        new_pts = pts + momentum
        new_pts = _project_to_sphere(new_pts)
        
        cr = _covering_radius(new_pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            lr *= 0.9
            momentum = mu * momentum
        
        if lr < 1e-6:
            break
    
    return best_pts, best_cr

def _optimize_full(points, deadline):
    """Full optimization pipeline with time management."""
    best_pts = points.copy()
    best_cr = _covering_radius(points)
    
    # Phase 1: Repulsion-based optimization
    if time.time() < deadline - 30:
        rep_pts, rep_cr = _optimize_repulsion(points, max_iters=100, lr=0.015, k=8)
        if rep_cr < best_cr:
            best_cr = rep_cr
            best_pts = rep_pts.copy()
    
    # Phase 2: Soft-max polish
    if time.time() < deadline - 10:
        soft_pts, soft_cr = _optimize_softmax_polish(best_pts, max_iters=100, lr=0.01, beta=25.0)
        if soft_cr < best_cr:
            best_cr = soft_cr
            best_pts = soft_pts.copy()
    
    # Phase 3: Final fine-tuning with smaller steps
    if time.time() < deadline - 2:
        lr = 0.005
        for it in range(50):
            if time.time() > deadline - 2:
                break
            forces = _repulsion_force(best_pts, k=6)
            tangent_forces = forces - np.sum(forces * best_pts, axis=1, keepdims=True) * best_pts
            new_pts = best_pts + lr * tangent_forces
            new_pts = _project_to_sphere(new_pts)
            cr = _covering_radius(new_pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = new_pts
            else:
                lr *= 0.9
    
    return best_pts, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    best_pts = None
    best_cr = np.pi
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral (good starting point)
    pts_fib = _fibonacci_sphere(N)
    initializations.append(pts_fib)
    
    # Perturbed Fibonacci
    for perturb_scale in [0.05, 0.1, 0.2]:
        pts_pert = pts_fib + perturb_scale * rng.standard_normal((N, 3))
        pts_pert = _project_to_sphere(pts_pert)
        initializations.append(pts_pert)
    
    # Random initializations
    for _ in range(2):
        pts_r = rng.standard_normal((N, 3))
        pts_r = _project_to_sphere(pts_r)
        initializations.append(pts_r)
    
    # Optimize each initialization
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        opt_pts, cr = _optimize_full(init_pts, deadline)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = opt_pts.copy()
    
    if best_pts is None:
        best_pts = initializations[0]
    
    return best_pts
# EVOLVE-BLOCK-END
