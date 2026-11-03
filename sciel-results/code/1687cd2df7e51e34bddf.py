"""Rewritten program for packing 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(pts):
    """Compute the largest common radius for given centers."""
    if len(pts) < 2:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    r = min(np.min(dist_boundary), min_pair_dist / 2.0)
    return max(r, 0.0)


def _fermat_spiral_init(n):
    """Generate points using Fermat's spiral for uniform distribution in unit disk."""
    i = np.arange(n)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = i * golden_angle
    r = np.sqrt(i / n) * 0.95  # Scale to stay well within unit circle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def _hex_init(n):
    """Generate points using hexagonal lattice."""
    # Estimate spacing
    area_per_circle = np.pi / n
    spacing = np.sqrt(area_per_circle * 2 / np.sqrt(3))
    
    points = []
    # Generate hex lattice points
    for j in range(-15, 16):
        for i in range(-15, 16):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3) / 2
            if x * x + y * y < 0.9:
                points.append([x, y])
    
    points = np.array(points)
    if len(points) >= n:
        # Sort by distance from center and take first n
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)
        return points[idx[:n]]
    else:
        return _fermat_spiral_init(n)


def _objective_and_grad(pts_flat, n):
    """
    Objective function: minimize negative of the minimum radius.
    We want to maximize min(min_pair_dist/2, min_boundary_dist).
    So we minimize -min(...).
    
    For smoothness, we use a soft-min approximation via log-sum-exp.
    """
    pts = pts_flat.reshape(n, 2)
    
    # Boundary distances
    dist_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-15)
    np.fill_diagonal(dist_matrix, np.inf)
    
    # For each pair, we care about dist/2
    # We want the minimum over all pairs and boundary
    # Use soft-min with log-sum-exp for smoothness
    
    # Collect all constraints: boundary distances and half pairwise distances
    # To keep it efficient, we work with the matrix
    
    # Soft-min of boundary distances
    beta = 50.0
    log_sum_exp_boundary = np.log(np.sum(np.exp(-beta * dist_boundary))) / (-beta)
    
    # For pairwise, we need min over all i<j of dist_ij/2
    # This is trickier. Let's use a different approach.
    # Actually, let's just compute the actual min and use a smoothed version.
    
    # Simpler: compute actual radius, then use a smooth penalty
    min_pair = np.min(dist_matrix)
    min_boundary = np.min(dist_boundary)
    r = min(min_pair / 2.0, min_boundary)
    
    # Gradient computation is complex. Let's use numerical gradient or
    # use scipy's minimize with numerical gradients.
    
    return -r  # We minimize negative radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial point
    pts = _hex_init(N)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Optimization using Nelder-Mead or Powell
    # We'll use a custom approach with local perturbations
    
    # Phase 1: Coarse optimization with Nelder-Mead
    def objective(pts_flat):
        pts = pts_flat.reshape(N, 2)
        # Check validity
        if np.any(~np.isfinite(pts)):
            return 1e10
        dist_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        if np.any(dist_boundary < -0.1):
            return 1e10
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dist_matrix = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-15)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        r = min(min_pair / 2.0, np.min(dist_boundary))
        return -r
    
    try:
        result = minimize(
            objective,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if result.fun < -best_r:
            best_r = -result.fun
            best_pts = result.x.reshape(N, 2)
    except Exception:
        pass
    
    if time.time() > deadline - 30.0:
        return best_pts
    
    # Phase 2: Local refinement - move each point slightly to improve
    # Use a simple iterative improvement
    for iteration in range(200):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 10.0:
                    break
            
            # Try small perturbations
            base_pt = best_pts[i].copy()
            base_r = best_r
            
            for dx in [-0.001, 0, 0.001]:
                for dy in [-0.001, 0, 0.001]:
                    if dx == 0 and dy == 0:
                        continue
                    test_pts = best_pts.copy()
                    test_pts[i] = base_pt + np.array([dx, dy])
                    r = _compute_radius(test_pts)
                    if r > best_r + 1e-12:
                        best_r = r
                        best_pts = test_pts
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            # Try smaller steps
            for dx in [-0.0001, 0.0001]:
                for dy in [-0.0001, 0.0001]:
                    if time.time() > deadline - 10.0:
                        break
                    test_pts = best_pts.copy()
                    test_pts[i] = base_pt + np.array([dx, dy])
                    r = _compute_radius(test_pts)
                    if r > best_r + 1e-12:
                        best_r = r
                        best_pts = test_pts
                        improved = True
                        break
    
    # Final validation
    if np.any(~np.isfinite(best_pts)):
        best_pts = _fermat_spiral_init(N)
    
    return best_pts

# EVOLVE-BLOCK-END
