"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import time


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses scipy.optimize.least_squares to maximize the minimum radius
    by minimizing a smooth proxy for the max-min problem.
    """
    n = N
    
    # Initial guess: hexagonal lattice points inside a slightly smaller disk
    # This provides a good starting point for the optimization.
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 0.95
    points = points[mask]
    
    # Ensure we have exactly n points
    if len(points) < n:
        # Fallback: add random points or expand grid
        # For now, just take what we have and pad if necessary, 
        # but the grid should be dense enough.
        # Let's generate more points if needed
        extra_spacing = spacing * 0.9
        k2 = int(1.0 / extra_spacing) + 3
        rows2 = [((i + 0.5 * (j % 2)) * extra_spacing, j * extra_spacing * np.sqrt(3.0) / 2.0)
                 for i in range(-k2, k2 + 1) for j in range(-k2, k2 + 1)]
        points2 = np.array(rows2)
        radius2 = np.hypot(points2[:, 0], points2[:, 1])
        points2 = points2[radius2 <= 0.95]
        points = np.vstack([points, points2])
        points = points[:n]
    else:
        points = points[:n]
        
    x0 = points.flatten()
    
    # Small epsilon to avoid division by zero
    delta = 1e-6
    # Scaling factor for pair residuals to balance with boundary residuals
    # There are ~n^2/2 pairs and n points.
    # We want the sum of squares to be comparable.
    # If typical residual is R, sum_pairs ~ n^2 R^2, sum_pts ~ n R^2.
    # So scale pairs by 1/sqrt(n).
    scale_pair = 1.0 / np.sqrt(n)
    # Penalty strength for being outside the boundary
    K_out = 100.0
    
    def residuals(x):
        centres = x.reshape(n, 2)
        
        # Compute pairwise distances
        dists = pdist(centres)
        
        # Pair residuals: 1 / (d + delta)
        # We want to minimize this, so maximize d.
        pair_res = 1.0 / (dists + delta)
        pair_res *= scale_pair
        
        # Boundary distances
        norms = np.linalg.norm(centres, axis=1)
        b = 1.0 - norms
        
        # Boundary residuals: 1 / (b + delta) + K * max(0, -b)^2
        # The first term pushes b to be large (inward).
        # The second term penalizes b < 0 (outside).
        # Note: max(0, -b)^2 is smooth.
        outside_penalty = np.maximum(0, -b) ** 2
        boundary_res = 1.0 / (b + delta) + K_out * outside_penalty
        
        return np.concatenate([pair_res, boundary_res])
    
    # Run least_squares
    # 'trf' is robust and handles the smoothness well
    result = least_squares(residuals, x0, method='trf', max_nfev=50000, 
                           xtol=1e-10, ftol=1e-10, gtol=1e-10)
    
    centres = result.x[:2*n].reshape(n, 2)
    
    # Compute the actual radius
    dists = pdist(centres)
    min_pair_dist = np.min(dists)
    boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    # Verify validity
    if r <= 0 or not np.isfinite(r) or np.any(boundary_dists < 0):
        # Fallback to a valid configuration if optimization failed
        # Use the initial hexagonal grid points, scaled to fit
        spacing = 2.0 / (np.sqrt(n) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points = points[radius <= 1.0 - spacing / 2.0]
            if len(points) >= n:
                return points[np.argsort(radius, kind="stable")[:n]]
            spacing *= 0.95
            if spacing < 0.01:
                 # Last resort: random
                 return np.random.uniform(-0.9, 0.9, (n, 2))
    
    return centres
# EVOLVE-BLOCK-END
