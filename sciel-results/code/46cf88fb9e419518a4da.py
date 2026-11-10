"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


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
    N = len(centers)
    
    if N < 2:
        return 0.5
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dist_sq))
    
    # Boundary distances
    # Distance to each face
    boundary_dists = np.minimum(centers + 0.5, 0.5 - centers)  # shape (N, 3)
    min_boundary = np.min(boundary_dists, axis=1)  # shape (N,)
    min_boundary_dist = np.min(min_boundary)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def construct():
    """
    Construct 50 sphere centers in the unit cube [-1/2, 1/2]^3 using a
    space-filling curve initialization followed by gradient-based optimization.
    """
    # Step 1: Generate multiple initializations using space-filling curves
    initializations = []
    
    # Morton curve with different grid sizes
    for n in [4, 5, 6, 7]:
        morton_pts = _morton3d_curve(n, N)
        # Map from [0,1)^3 to [-1/2, 1/2]^3
        morton_pts = morton_pts - 0.5
        initializations.append(morton_pts.copy())
    
    # Grid-based initialization with multiple offsets
    np.random.seed(42)
    for m in [4, 5]:
        ticks = (np.arange(m) + 0.5) / m - 0.5
        all_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        perm = np.random.permutation(len(all_points))
        grid_init = all_points[perm[:N]].copy()
        initializations.append(grid_init)
    
    # Random initialization
    for _ in range(3):
        rand_init = np.random.uniform(-0.4, 0.4, size=(N, 3))
        initializations.append(rand_init)
    
    # Step 2: Optimize each initialization
    best_centers = None
    best_radius = -1.0
    
    # Objective function: minimize sum of inverse squared distances (smooth)
    def objective(x):
        """
        x: flat array of shape (N*3,)
        Returns a scalar objective to minimize.
        We want to maximize the minimum pairwise distance and boundary distance.
        Use sum of inverse squared distances as a smooth proxy.
        """
        pts = x.reshape(N, 3)
        
        # Pairwise distances (squared)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        # Sum of inverse squared distances for pairs
        # Clip to avoid division by zero
        dist_sq_clipped = np.maximum(dist_sq, 1e-12)
        pair_obj = np.sum(1.0 / dist_sq_clipped)
        
        # Boundary distances (squared)
        # Distance to each face
        boundary_dists = np.minimum(pts + 0.5, 0.5 - pts)  # shape (N, 3)
        boundary_dists_sq = np.minimum(boundary_dists ** 2, 0.25)
        boundary_dists_sq_clipped = np.maximum(boundary_dists_sq, 1e-12)
        boundary_obj = np.sum(1.0 / boundary_dists_sq_clipped)
        
        # Combined objective: we want both to be large, so minimize their inverses
        # Weight: pairwise distances are more critical
        obj = 3.0 * pair_obj + 1.0 * boundary_obj
        
        return obj
    
    # Gradient of objective (for L-BFGS-B efficiency) - vectorized
    def gradient(x):
        pts = x.reshape(N, 3)
        
        # Pairwise gradient (vectorized)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist_sq_clipped = np.maximum(dist_sq, 1e-12)
        
        # d(1/d^2)/d(pts_i) = 2 * sum_j (pts_i - pts_j) / d_ij^4
        # For minimization of sum(1/d^2), gradient is positive (push apart)
        inv_d4 = 1.0 / (dist_sq_clipped ** 2)
        np.fill_diagonal(inv_d4, 0.0)
        grad_pair = 2.0 * np.sum(diffs * inv_d4[:, :, np.newaxis], axis=1)
        
        # Boundary gradient (vectorized)
        boundary_dists = np.minimum(pts + 0.5, 0.5 - pts)
        boundary_dists_sq = np.minimum(boundary_dists ** 2, 0.25)
        boundary_dists_sq_clipped = np.maximum(boundary_dists_sq, 1e-12)
        
        # Determine which face is closer
        closer_to_neg = (pts + 0.5) <= (0.5 - pts)
        signs = np.where(closer_to_neg, 1.0, -1.0)
        
        # d(1/bd^2)/d(pts_ik) = -sign / bd^3
        inv_bd3 = signs / (boundary_dists_sq_clipped ** 1.5)
        grad_boundary = -inv_bd3
        
        grad = 3.0 * grad_pair + 1.0 * grad_boundary
        return grad.flatten()
    
    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3)
    
    for init in initializations:
        x0 = init.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
            )
            
            centers = result.x.reshape(N, 3)
            centers = np.clip(centers, -0.5 + 1e-8, 0.5 - 1e-8)
            
            radius = _compute_radius(centers)
            
            if radius > best_radius:
                best_radius = radius
                best_centers = centers.copy()
                
        except Exception:
            continue
    
    # If no optimization succeeded, use the best initialization
    if best_centers is None:
        best_centers = initializations[0]
        best_radius = _compute_radius(best_centers)
    
    return best_centers

# EVOLVE-BLOCK-END
