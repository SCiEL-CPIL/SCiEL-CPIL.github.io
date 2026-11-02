"""Differentiable rigid-body mechanics approach for circle packing in a unit disk."""
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed and select n points closest to center."""
    # For n circles in unit disk, estimate spacing
    # Area per circle ~ pi/n, radius ~ sqrt(1/n)
    # Hexagonal spacing ~ 2*r * 1.15 (accounting for boundary effects)
    r_est = np.sqrt(1.0 / (n * 1.2))
    spacing = 2.0 * r_est * 1.15
    
    # Generate hexagonal lattice points
    points = []
    k_max = int(1.0 / spacing) + 5
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    
    # Keep points that fit (with some margin)
    margin = spacing * 0.1
    mask = radii <= 1.0 - margin
    points = points[mask]
    radii = radii[mask]
    
    if len(points) < n:
        # If not enough, reduce spacing
        for _ in range(30):
            spacing *= 0.95
            points = []
            k_max = int(1.0 / spacing) + 5
            for j in range(-k_max, k_max + 1):
                for i in range(-k_max, k_max + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    points.append([x, y])
            points = np.array(points)
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii <= 1.0 - margin
            points = points[mask]
            radii = radii[mask]
            if len(points) >= n:
                break
    
    # Sort by distance from center and take first n
    idx = np.argsort(radii)
    return points[idx[:n]].copy()


def _compute_score(centers):
    """Compute the packing radius for given centers using vectorized operations."""
    n = len(centers)
    
    # Compute all pairwise distances
    # Use broadcasting for efficiency
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Get minimum off-diagonal distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Minimum distance to boundary
    min_boundary = np.min(1.0 - np.linalg.norm(centers, axis=1))
    
    return min(min_pair_dist / 2.0, min_boundary)


def _objective_and_grad(x, n, target_r, use_log=False):
    """
    Compute objective and gradient for maximizing minimum clearance.
    We minimize the negative of the minimum clearance, but use a smooth approximation.
    
    Strategy: minimize sum of squared violations for pairs closer than 2*target_r
    and boundary violations, then the minimum clearance will be at least target_r.
    """
    c = x.reshape(n, 2)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    
    # Boundary distances
    boundary_dists = 1.0 - np.linalg.norm(c, axis=1)
    
    target_d = 2.0 * target_r
    
    # Pairwise violations: penalize if dist < target_d
    # Use smooth approximation: for pairs where dist < target_d, add (target_d - dist)^2
    pair_violation = np.maximum(0, target_d - dists)
    # Only consider i < j to avoid double counting
    iu = np.triu_indices(n, k=1)
    pair_violation_ij = pair_violation[iu]
    pair_obj = np.sum(pair_violation_ij ** 2)
    
    # Boundary violations: penalize if boundary_dist < target_r
    boundary_violation = np.maximum(0, target_r - boundary_dists)
    boundary_obj = np.sum(boundary_violation ** 2)
    
    obj = pair_obj + boundary_obj
    
    # Gradient computation
    grad = np.zeros_like(x)
    
    # Pairwise gradient
    # For each pair (i,j) with dist < target_d:
    # d/d(c_i) [(target_d - dist_ij)^2] = -2*(target_d - dist_ij) * (c_i - c_j)/dist_ij
    # d/d(c_j) [(target_d - dist_ij)^2] = -2*(target_d - dist_ij) * (c_j - c_i)/dist_ij
    
    # Only compute for violating pairs
    violating_pairs = pair_violation > 1e-12
    violating_pairs[0, 0] = False  # diagonal
    violating_pairs = violating_pairs & (np.arange(n)[:, None] < np.arange(n)[None, :])
    
    if np.any(violating_pairs):
        # Get the violating pair indices
        vi, vj = np.where(violating_pairs)
        
        for i, j in zip(vi, vj):
            d_ij = dists[i, j]
            if d_ij < 1e-10:
                continue
            factor = -2.0 * (target_d - d_ij) / d_ij
            diff_ij = c[i] - c[j]
            grad[2*i:2*i+2] += factor * diff_ij
            grad[2*j:2*j+2] -= factor * diff_ij
    
    # Boundary gradient
    # For each circle with boundary_dist < target_r:
    # d/d(c_i) [(target_r - (1 - |c_i|))^2] = -2*(target_r - (1-|c_i|)) * c_i/|c_i|
    
    violating_boundary = boundary_violation > 1e-12
    if np.any(violating_boundary):
        vb = np.where(violating_boundary)[0]
        for i in vb:
            norm_i = np.linalg.norm(c[i])
            if norm_i < 1e-10:
                continue
            factor = -2.0 * (target_r - boundary_dists[i]) / norm_i
            grad[2*i:2*i+2] += factor * c[i]
    
    return obj, grad


def _optimize_for_radius(centers, target_r, max_iter=500):
    """
    Optimize centers so that all pairwise distances >= 2*target_r
    and all circles are within radius (1 - target_r) from center.
    """
    n = len(centers)
    x0 = centers.flatten()
    
    result = minimize(
        _objective_and_grad,
        x0,
        args=(n, target_r),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    return result.x.reshape(n, 2)


def _check_feasible(centers, target_r, tol=1e-7):
    """Check if the configuration is feasible for the given target radius."""
    n = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary
    min_boundary = np.min(1.0 - np.linalg.norm(centers, axis=1))
    
    return min(min_pair / 2.0, min_boundary) >= target_r - tol


def construct():
    """
    Construct 300 equal circles in a unit circle, maximizing the common radius.
    Uses differentiable rigid-body mechanics with spring constraints.
    """
    # Generate initial hexagonal seed
    centers = _hexagonal_seed(N)
    
    # Estimate initial radius
    initial_score = _compute_score(centers)
    
    # Binary search on the radius
    low = initial_score * 0.85
    high = initial_score * 1.05
    
    best_centers = centers.copy()
    best_radius = initial_score
    
    for iteration in range(25):
        mid = (low + high) / 2.0
        
        # Try to find a valid configuration for radius mid
        optimized = _optimize_for_radius(centers, mid, max_iter=300)
        
        # Check if the optimized configuration is valid
        score = _compute_score(optimized)
        
        if score >= mid - 1e-6:
            # Feasible, try larger radius
            low = mid
            best_centers = optimized.copy()
            best_radius = score
            # Use optimized as new starting point
            centers = optimized.copy()
        else:
            # Not feasible, try smaller radius
            high = mid
        
        if high - low < 1e-7:
            break
    
    # Final verification and cleanup
    final_score = _compute_score(best_centers)
    
    # Ensure all points are within the unit circle
    for i in range(len(best_centers)):
        r = np.linalg.norm(best_centers[i])
        if r > 1.0:
            scale = (1.0 - 1e-10) / r
            best_centers[i] *= scale
    
    return best_centers
# EVOLVE-BLOCK-END
