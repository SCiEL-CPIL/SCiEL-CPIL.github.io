"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _hilbert3d(i, n):
    """
    Convert integer i in [0, n^3) to 3D coordinates in [0, n)^3
    using a 3D Hilbert curve.
    
    Uses the standard 3D Hilbert curve algorithm with rotations.
    """
    bits = int(np.log2(n))
    if n != (1 << bits):
        while (1 << bits) < n:
            bits += 1
    
    x, y, z = 0, 0, 0
    t = i
    
    for p in range(bits - 1, -1, -1):
        rx = (t >> 2) & 1
        ry = (t >> 1) & 1
        rz = t & 1
        
        if rz == 0:
            if ry == 0:
                x, z = z, x
            if rx == 0:
                y, z = z, y
        
        if rx == 1:
            x = (1 << p) - 1 - x
        if ry == 1:
            y = (1 << p) - 1 - y
        
        x |= rx << p
        y |= ry << p
        z |= rz << p
        
        t >>= 3
    
    return np.array([x, y, z], dtype=float)


def _hilbert3d_curve(n, count):
    """
    Generate 'count' points from a 3D Hilbert curve of size n^3.
    Returns points in [0, 1)^3.
    """
    points = []
    for i in range(count):
        p = _hilbert3d(i, n)
        points.append(p / n)
    return np.array(points)


def _morton3d(i, bits):
    """Compute 3D Morton code for a 1D index."""
    x = 0
    y = 0
    z = 0
    for b in range(bits):
        x |= ((i >> (3 * b)) & 1) << b
        y |= ((i >> (3 * b + 1)) & 1) << b
        z |= ((i >> (3 * b + 2)) & 1) << b
    return x, y, z


def _morton3d_curve(n, count):
    """
    Generate 'count' points from a 3D Morton (Z-order) curve of size n^3.
    Returns points in [0, 1)^3.
    """
    bits = int(np.log2(n))
    points = []
    for i in range(count):
        x, y, z = _morton3d(i, bits)
        points.append(np.array([x, y, z], dtype=float) / n)
    return np.array(points)


def _compute_radius(centers):
    """
    Compute the maximum common radius for given centers in [-1/2, 1/2]^3.
    """
    centers = np.asarray(centers)
    n = len(centers)
    
    if n < 2:
        return 0.5
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Boundary distances
    boundary_dists = np.minimum(centers + 0.5, 0.5 - centers)
    min_boundary = np.min(boundary_dists, axis=1)
    min_boundary_dist = np.min(min_boundary)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _objective_and_grad(x, n_points):
    """
    Compute objective and gradient for the sphere packing problem.
    Uses a smooth approximation: minimize sum of 1/d^2 for all pairs and boundaries.
    """
    pts = x.reshape(n_points, 3)
    
    # Pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Clip to avoid numerical issues
    eps = 1e-10
    dist_sq_safe = np.maximum(dist_sq, eps)
    
    # Objective: sum of 1/d^2 for pairs
    pair_obj = np.sum(1.0 / dist_sq_safe)
    
    # Gradient for pairwise part
    # d(1/d^2)/d(pts_i) = -2 * (pts_i - pts_j) / d^4
    grad_pair = np.zeros_like(pts)
    # Vectorized computation
    inv_d4 = 1.0 / (dist_sq_safe ** 2)
    np.fill_diagonal(inv_d4, 0.0)
    grad_pair = -2.0 * np.sum(diffs * inv_d4[:, :, np.newaxis], axis=1)
    
    # Boundary distances
    # For each point and each dimension, distance to nearest face
    # dist_to_face = min(pt + 0.5, 0.5 - pt)
    boundary_dists = np.minimum(pts + 0.5, 0.5 - pts)
    boundary_dists_safe = np.maximum(boundary_dists, eps)
    
    # Objective: sum of 1/bd^2 for boundaries
    boundary_obj = np.sum(1.0 / (boundary_dists_safe ** 2))
    
    # Gradient for boundary part
    # For each point i and dimension k:
    # if pts[i,k] < 0: closer to -0.5 face, bd = pts[i,k] + 0.5, d(bd)/d(pts) = 1
    # if pts[i,k] > 0: closer to 0.5 face, bd = 0.5 - pts[i,k], d(bd)/d(pts) = -1
    # d(1/bd^2)/d(pts_ik) = -2/bd^3 * d(bd)/d(pts_ik)
    
    sign = np.where(pts <= 0.0, 1.0, -1.0)
    grad_boundary = -2.0 * sign / (boundary_dists_safe ** 3)
    
    # Combine: weight pairwise more heavily
    weight_pair = 10.0
    weight_boundary = 1.0
    
    obj = weight_pair * pair_obj + weight_boundary * boundary_obj
    grad = weight_pair * grad_pair + weight_boundary * grad_boundary
    
    return obj, grad.flatten()


def _optimize_from_start(x0, n_points, maxiter=300):
    """
    Run L-BFGS-B optimization from a given starting point.
    Returns (centers, radius).
    """
    bounds = [(-0.5 + 1e-8, 0.5 - 1e-8)] * (n_points * 3)
    
    try:
        result = minimize(
            lambda x: _objective_and_grad(x, n_points)[0],
            x0,
            jac=lambda x: _objective_and_grad(x, n_points)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': maxiter,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxfun': 10000,
                'maxls': 20
            }
        )
        
        centers = result.x.reshape(n_points, 3)
        centers = np.clip(centers, -0.5 + 1e-8, 0.5 - 1e-8)
        radius = _compute_radius(centers)
        
        return centers, radius
        
    except Exception:
        # If optimization fails, return the starting point
        centers = x0.reshape(n_points, 3)
        centers = np.clip(centers, -0.5 + 1e-8, 0.5 - 1e-8)
        radius = _compute_radius(centers)
        return centers, radius


def construct():
    """
    Construct 50 sphere centers in the unit cube [-1/2, 1/2]^3 using a
    space-filling curve initialization followed by gradient-based optimization.
    """
    initializations = []
    
    # Generate initializations from different space-filling curves
    # Hilbert curves with different resolutions
    for n in [4, 5, 6, 7]:
        hilbert_pts = _hilbert3d_curve(n, N)
        hilbert_pts = hilbert_pts - 0.5
        initializations.append(hilbert_pts.copy())
        
        # Morton curves
        morton_pts = _morton3d_curve(n, N)
        morton_pts = morton_pts - 0.5
        initializations.append(morton_pts.copy())
    
    # Add a structured grid initialization
    # Use a 3x3x3 grid (27 points) and add 23 more by subdividing
    base_ticks = np.array([-0.333, 0.0, 0.333])
    grid_pts = np.array([(x, y, z) for z in base_ticks for y in base_ticks for x in base_ticks])
    
    # Add more points by using a finer grid for the remaining
    fine_ticks = np.linspace(-0.45, 0.45, 5)
    fine_grid_pts = np.array([(x, y, z) for z in fine_ticks for y in fine_ticks for x in fine_ticks])
    
    # Combine and take first N
    all_grid_pts = np.vstack([grid_pts, fine_grid_pts])
    np.random.seed(123)
    perm = np.random.permutation(len(all_grid_pts))
    grid_init = all_grid_pts[perm[:N]].copy()
    initializations.append(grid_init)
    
    # Add a random initialization for diversity
    np.random.seed(456)
    random_init = np.random.uniform(-0.45, 0.45, (N, 3))
    initializations.append(random_init)
    
    # Optimize from each initialization and keep the best
    best_centers = None
    best_radius = -1.0
    
    for init in initializations:
        centers, radius = _optimize_from_start(init.flatten(), N, maxiter=250)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # If no optimization succeeded, use the best initialization
    if best_centers is None:
        best_centers = initializations[0]
        best_radius = _compute_radius(best_centers)
    
    # Final local refinement: try small perturbations around the best solution
    # This can help escape shallow local minima
    for _ in range(3):
        perturbation = np.random.normal(0, 0.01, (N, 3))
        perturbed = best_centers + perturbation
        perturbed = np.clip(perturbed, -0.5 + 1e-8, 0.5 - 1e-8)
        
        centers, radius = _optimize_from_start(perturbed.flatten(), N, maxiter=100)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    return best_centers

# EVOLVE-BLOCK-END
