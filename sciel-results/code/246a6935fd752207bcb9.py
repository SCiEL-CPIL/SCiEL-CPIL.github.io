# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    n_pairs = n * (n - 1) // 2

    # Precompute pair indices
    i_idx = []
    j_idx = []
    for i in range(n):
        for j in range(i + 1, n):
            i_idx.append(i)
            j_idx.append(j)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)

    def compute_ratio(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return d_min / d_max

    def compute_ratio_squared(points):
        """Return R = (d_min/d_max)^2, which is the actual metric."""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Generate diverse initial configurations
    initial_points = []

    # 4x4 grid
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    initial_points.append(grid.flatten())

    # Points on a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_points.append(circle.flatten())

    # Hexagonal pattern
    hex_points = []
    hex_points.append([0, 0])
    for k in range(6):
        angle = k * np.pi / 3
        hex_points.append([np.cos(angle), np.sin(angle)])
    for k in range(12):
        angle = k * np.pi / 6
        r = 2
        hex_points.append([r * np.cos(angle), r * np.sin(angle)])
    hex_points = hex_points[:n]
    initial_points.append(np.array(hex_points).flatten())

    # Two concentric circles
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    inner_r = 0.5
    outer_r = 1.0
    inner_pts = inner_r * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    outer_pts = outer_r * np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    two_circles = np.vstack([inner_pts, outer_pts])
    initial_points.append(two_circles.flatten())

    # Random starts with fixed seed
    np.random.seed(42)
    for _ in range(20):
        initial_points.append(np.random.randn(n * 2))

    # Optimize using SLSQP on the ratio objective
    def objective(x):
        points = x.reshape(n, 2)
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return -(d_min / d_max)

    best_points = None
    best_score = -np.inf

    for x0 in initial_points:
        # Normalize initial point to unit scale
        x0_norm = x0.copy()
        pts0 = x0_norm.reshape(n, 2)
        d_max0 = np.max(pdist(pts0))
        if d_max0 > 0:
            x0_norm = x0_norm / d_max0

        result = minimize(objective, x0_norm, method='SLSQP', options={'maxiter': 5000, 'ftol': 1e-14})

        if result.success:
            points = result.x.reshape(n, 2)
            score = compute_ratio(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()

    if best_points is None:
        best_points = initial_points[0].reshape(n, 2)

    # Now perform bisection on the minimum distance r, with max distance constrained to 1
    # For a given r, try to find 16 points with all pairwise distances >= r and <= 1
    # This is a feasibility problem. We use SQP to maximize the minimum distance
    # subject to max distance <= 1.

    def feasibility_objective(x, r_target):
        """Maximize min distance - penalty for exceeding max distance."""
        points = x.reshape(n, 2)
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        # We want d_min >= r_target and d_max <= 1
        # Objective: maximize (d_min - r_target) - penalty * max(0, d_max - 1)
        penalty = 100.0
        obj = (d_min - r_target) - penalty * max(0.0, d_max - 1.0)
        return -obj

    # Bisection loop: find the largest r such that feasible configuration exists
    r_low = 0.0
    r_high = 1.0

    # First, verify that r_high is infeasible and r_low is feasible
    # Start with the best_points we found, normalize to max dist = 1
    if best_points is not None:
        bp = best_points.copy()
        d_max_bp = np.max(pdist(bp))
        if d_max_bp > 0:
            bp = bp / d_max_bp
        d_min_bp = np.min(pdist(bp))
        r_low = max(0.0, d_min_bp * 0.5)
        r_high = min(1.0, d_min_bp * 1.5)

    # Bisection iterations
    best_r = r_low
    best_config = best_points.copy() if best_points is not None else np.zeros((n, 2))

    for iteration in range(30):
        r_mid = (r_low + r_high) / 2.0

        # Try to find a feasible configuration for r_mid
        # Use multiple starts
        found_feasible = False
        best_x_for_r = None
        best_obj_for_r = np.inf

        starts_for_bisection = []
        # Use best_config as a start
        if best_config is not None:
            starts_for_bisection.append(best_config.flatten())
        # Add some random starts
        np.random.seed(42 + iteration)
        for _ in range(5):
            starts_for_bisection.append(np.random.randn(n * 2))

        for x0 in starts_for_bisection:
            x0_norm = x0.copy()
            pts0 = x0_norm.reshape(n, 2)
            d_max0 = np.max(pdist(pts0))
            if d_max0 > 0:
                x0_norm = x0_norm / d_max0

            res = minimize(feasibility_objective, x0_norm, args=(r_mid,), method='SLSQP',
                           options={'maxiter': 2000, 'ftol': 1e-12})

            if res.success:
                pts = res.x.reshape(n, 2)
                dists = pdist(pts)
                d_min = np.min(dists)
                d_max = np.max(dists)
                # Check feasibility: d_min >= r_mid and d_max <= 1 (with tolerance)
                tol = 1e-6
                if d_min >= r_mid - tol and d_max <= 1.0 + tol:
                    found_feasible = True
                    if res.fun < best_obj_for_r:
                        best_obj_for_r = res.fun
                        best_x_for_r = res.x.copy()

        if found_feasible:
            r_low = r_mid
            if best_x_for_r is not None:
                best_r = r_mid
                best_config = best_x_for_r.reshape(n, 2)
        else:
            r_high = r_mid

    # Use the best configuration found
    if best_config is not None:
        # Verify the final score
        final_score = compute_ratio(best_config)
        if final_score > best_score:
            best_points = best_config
            best_score = final_score

    if best_points is None:
        best_points = initial_points[0].reshape(n, 2)

    return best_points

# EVOLVE-BLOCK-END
