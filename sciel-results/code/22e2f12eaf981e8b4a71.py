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
    Uses scipy.optimize.least_squares to maximize the minimum radius.
    Variables: x = [x_1, y_1, ..., x_n, y_n, r]
    Residuals:
    - For each pair (i, j): 2*r - d_ij
    - For each point i: r - (1 - |c_i|)
    """
    n = N
    
    # 1. Initial Guess: Hexagonal lattice
    # Estimate spacing
    # Area of unit circle = pi. Area per circle ~ pi/300 ~ 0.0105.
    # Hex packing density ~ 0.9069.
    # Effective area per circle ~ 0.0105 / 0.9069 ~ 0.0116.
    # Hex unit cell area = sqrt(3)/2 * s^2.
    # s^2 ~ 0.0116 / 0.866 ~ 0.0134 => s ~ 0.116.
    
    spacing = 0.115
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 0.95
    points = points[mask]
    
    if len(points) < n:
        # Fallback: Random or Grid if hex fails (unlikely)
        # Use a dense grid
        xs = np.linspace(-0.95, 0.95, 25)
        ys = np.linspace(-0.95, 0.95, 25)
        grid = np.array([[x, y] for x in xs for y in ys])
        # Filter to inside
        r_grid = np.hypot(grid[:, 0], grid[:, 1])
        grid = grid[r_grid <= 0.95]
        points = grid[:n]
    else:
        # Sort by radius to pick central ones? Or just first n?
        # Taking first n might be biased. Let's take the ones closest to the center
        # to ensure a dense core, which is usually better for packing.
        # Actually, for packing, we want to fill the whole disk.
        # The hex grid generation above creates a disk-shaped set of points.
        # We should select n points.
        # If we have more than n, we can randomly sample or take a subset.
        # Taking a subset that preserves the structure is hard.
        # Let's just take the first n after sorting by radius?
        # Or better, since the hex grid is roughly uniform, we can just take n points.
        # To be safe, let's sort by radius and take the first n (innermost).
        # This might leave the outer ring empty, which is bad for boundary constraints.
        # Instead, let's generate a hex grid that has roughly n points.
        # Adjust spacing to get close to n points.
        pass

    # Let's refine the initial guess generation to be more robust
    def generate_hex_points(n_target):
        # Try different spacings to get close to n_target points inside radius 0.98
        best_points = None
        best_diff = float('inf')
        for s in np.linspace(0.08, 0.15, 20):
            k = int(1.0 / s) + 2
            rows = [((i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            pts = np.array(rows)
            rad = np.hypot(pts[:, 0], pts[:, 1])
            pts = pts[rad <= 0.98]
            if len(pts) >= n_target:
                diff = abs(len(pts) - n_target)
                if diff < best_diff:
                    best_diff = diff
                    best_points = pts
        if best_points is None:
            # Fallback
            xs = np.linspace(-0.95, 0.95, 25)
            ys = np.linspace(-0.95, 0.95, 25)
            grid = np.array([[x, y] for x in xs for y in ys])
            r_grid = np.hypot(grid[:, 0], grid[:, 1])
            grid = grid[r_grid <= 0.95]
            return grid[:n_target]
        
        # Select n_target points from best_points
        # To maintain distribution, we can just take the first n_target?
        # Or randomly? Random might break symmetry.
        # Let's sort by radius and take first n_target?
        # This creates a solid disk of points, which is good.
        r_pts = np.hypot(best_points[:, 0], best_points[:, 1])
        idx = np.argsort(r_pts)
        return best_points[idx[:n_target]]

    points = generate_hex_points(n)
    
    # Initial radius estimate
    dists = pdist(points)
    min_pair_dist = np.min(dists)
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    r0 = min(min_pair_dist / 2.0, min_boundary_dist)
    
    # Variable vector: [x1, y1, ..., xn, yn, r]
    x0 = np.concatenate([points.flatten(), [r0]])
    
    def residuals(x):
        pts = x[:2*n].reshape(n, 2)
        r = x[-1]
        
        # Pairwise distances
        # pdist is efficient
        d = pdist(pts)
        # Residuals for pairs: 2*r - d
        # We want 2*r <= d, so 2*r - d <= 0
        # Minimizing square of this pushes 2*r towards d
        res_pairs = 2 * r - d
        
        # Boundary distances
        # 1 - |c_i|
        dist_to_center = np.linalg.norm(pts, axis=1)
        boundary_dist = 1.0 - dist_to_center
        # Residuals for boundary: r - boundary_dist
        # We want r <= boundary_dist, so r - boundary_dist <= 0
        res_boundary = r - boundary_dist
        
        return np.concatenate([res_pairs, res_boundary])
    
    # Optimization
    # Use trf for robustness
    try:
        result = least_squares(residuals, x0, method='trf', max_nfev=5000, 
                               xtol=1e-10, ftol=1e-10, gtol=1e-10)
        x_opt = result.x
        pts_opt = x_opt[:2*n].reshape(n, 2)
        r_opt = x_opt[-1]
        
        # Verify and potentially refine
        # The optimized r is a lower bound on the true max r if constraints are satisfied.
        # But we just return the points. The evaluator computes the score.
        
        # Check for validity
        if not np.all(np.isfinite(pts_opt)):
            raise ValueError("Non-finite values")
            
        # Optional: Second pass with higher precision or different method?
        # Given time limits, one good pass is usually sufficient.
        
        return pts_opt
        
    except Exception as e:
        # Fallback to initial guess
        return points

# EVOLVE-BLOCK-END
