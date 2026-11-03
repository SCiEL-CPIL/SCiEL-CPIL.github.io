"""Optimized packing of N=300 equal circles in the unit disk using scipy least_squares."""
import numpy as np
import time
import scipy.optimize as so

N = 300


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill

    # Generate initial positions using hexagonal lattice
    points = _hex_init()

    # Ensure we have exactly N points
    if len(points) > N:
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        points = points[idx]
    elif len(points) < N:
        # Add random points in the disk
        while len(points) < N:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.sqrt(np.random.uniform(0, 1))
            points = np.vstack([points, [[r * np.cos(angle), r * np.sin(angle)]]])

    # Flatten to 1D array for optimization
    x0 = points.flatten()

    # Define the least squares objective
    def residuals(x):
        """Compute residuals for the packing problem.
        
        We want to maximize the minimum radius r such that:
        - All pairwise distances >= 2r
        - All points are within distance (1-r) from origin
        
        We minimize violations of these constraints.
        """
        pts = x.reshape(-1, 2)
        n = len(pts)
        
        # Compute all pairwise distances
        # Use broadcasting for efficiency
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Get upper triangle indices (i < j)
        i_idx, j_idx = np.triu_indices(n, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        # Boundary distances
        radii = np.sqrt(np.sum(pts ** 2, axis=1))
        boundary_dists = 1.0 - radii
        
        # Current minimum radius estimate
        min_pair = np.min(pair_dists) / 2.0
        min_bound = np.min(boundary_dists)
        r = min(min_pair, min_bound)
        
        # Residuals: violations of constraints
        # For pairs: want dist >= 2r, so residual = 2r - dist (positive if violated)
        pair_residuals = 2.0 * r - pair_dists
        
        # For boundary: want radius <= 1-r, so residual = radius - (1-r) = radius + r - 1
        boundary_residuals = radii - (1.0 - r)
        
        # Combine residuals with weighting
        # Weight boundary violations more heavily to keep points inside
        all_residuals = np.concatenate([pair_residuals, 10.0 * boundary_residuals])
        
        return all_residuals

    # Define Jacobian (gradient) for faster convergence
    def jacobian(x):
        """Compute Jacobian of residuals."""
        pts = x.reshape(-1, 2)
        n = len(pts)
        
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
        
        # Get upper triangle indices
        i_idx, j_idx = np.triu_indices(n, k=1)
        pair_dists = dists[i_idx, j_idx]
        
        # Boundary distances
        radii = np.sqrt(np.sum(pts ** 2, axis=1))
        boundary_dists = 1.0 - radii
        
        # Current minimum radius estimate
        min_pair = np.min(pair_dists) / 2.0
        min_bound = np.min(boundary_dists)
        r = min(min_pair, min_bound)
        
        # Build Jacobian matrix
        # Shape: (n_pairs + n, 2*n)
        n_pairs = len(i_idx)
        jac = np.zeros((n_pairs + n, 2 * n))
        
        # Pairwise distance gradients
        # d(dist_ij)/d(x_i) = (x_i - x_j) / dist_ij
        # d(dist_ij)/d(x_j) = (x_j - x_i) / dist_ij
        for k in range(n_pairs):
            i = i_idx[k]
            j = j_idx[k]
            d = diff[i, j]  # x_i - x_j
            dist = dists[i, j]
            if dist > 1e-10:
                # Residual is 2r - dist, so d(residual)/d(x_i) = -d(dist)/d(x_i)
                jac[k, 2*i:2*i+2] = -d / dist
                jac[k, 2*j:2*j+2] = d / dist
        
        # Boundary distance gradients
        # Residual is radius - (1-r) = radius + r - 1
        # d(residual)/d(x_i) = x_i / radius
        for i in range(n):
            rad = radii[i]
            if rad > 1e-10:
                jac[n_pairs + i, 2*i:2*i+2] = 10.0 * pts[i] / rad
        
        return jac

    # Optimize using least_squares
    try:
        result = so.least_squares(
            residuals,
            x0,
            jac=jacobian,
            method='trf',  # Trust-region reflective
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=5000,
            bounds=([-1.0, -1.0] * N, [1.0, 1.0] * N)
        )
        if time.time() - start_time < time_limit:
            x_opt = result.x
        else:
            x_opt = x0
    except Exception as e:
        x_opt = x0

    # Convert back to points
    points_opt = x_opt.reshape(-1, 2)

    # Final validation and fix: ensure all points are inside unit circle
    radii = np.hypot(points_opt[:, 0], points_opt[:, 1])
    for i in range(len(points_opt)):
        if radii[i] > 1.0:
            scale = 0.999 / radii[i]
            points_opt[i] *= scale

    # Quick local polish with small perturbations
    if time.time() - start_time < time_limit - 10:
        points_opt = _polish(points_opt, start_time, time_limit)

    return points_opt


def _hex_init():
    """Generate hexagonal lattice points inside unit disk."""
    # Try different spacings to get approximately N points
    best_points = None
    best_count = 0
    
    for spacing in np.linspace(0.04, 0.08, 20):
        k = int(1.2 / spacing) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        
        if len(points) > best_count:
            best_count = len(points)
            best_points = np.array(points)
    
    if best_points is None:
        # Fallback: random
        points = []
        for _ in range(N):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.sqrt(np.random.uniform(0, 1))
            points.append([r * np.cos(angle), r * np.sin(angle)])
        return np.array(points)
    
    return best_points


def _polish(points, start_time, time_limit):
    """Quick local polish with small random perturbations."""
    best_points = points.copy()
    best_score = _score(points)
    
    n = len(points)
    step_size = 0.005
    
    for iteration in range(5000):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Randomly select a point
        i = np.random.randint(n)
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        new_point = points[i] + step_size * direction
        
        # Check if new position is valid
        if np.hypot(new_point[0], new_point[1]) >= 0.999:
            continue
        
        old_score = _score(points)
        points[i] = new_point
        new_score = _score(points)
        
        if new_score > best_score:
            best_points = points.copy()
            best_score = new_score
            step_size *= 0.999
        else:
            points[i] = new_point - step_size * direction
            step_size *= 0.995
    
    return best_points


def _score(points):
    """Compute the packing score."""
    n = len(points)
    if n < 2:
        return 0.0
    
    # Minimum pairwise distance
    min_pair_dist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair_dist:
                min_pair_dist = d
    
    # Minimum distance to boundary
    min_boundary = float('inf')
    for i in range(n):
        r = np.hypot(points[i, 0], points[i, 1])
        dist_to_boundary = 1.0 - r
        if dist_to_boundary < min_boundary:
            min_boundary = dist_to_boundary
    
    return min(min_pair_dist / 2.0, min_boundary)
# EVOLVE-BLOCK-END
