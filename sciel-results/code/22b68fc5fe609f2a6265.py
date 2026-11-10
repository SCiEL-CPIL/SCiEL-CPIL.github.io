# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600.0  # seconds, with safety margin before 630s kill

    # Helper: compute the metric R = (d_min / d_max)^2
    def compute_score(pts):
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Helper: normalize points (center at origin, scale so max distance is 1)
    def normalize_points(pts):
        pts = pts.copy()
        pts -= np.mean(pts, axis=0)
        # Scale so that the max pairwise distance is 1
        if n > 1:
            dists = pdist(pts)
            d_max = np.max(dists)
            if d_max > 0:
                pts /= d_max
        return pts

    # Generate initial candidate configurations
    candidates = []

    # 1. 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    candidates.append(normalize_points(grid))

    # 2. Hexagonal packing pattern
    # Place points in a hexagonal lattice pattern
    hex_pts = []
    # Center point
    hex_pts.append([0.0, 0.0])
    # First ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        hex_pts.append([np.cos(angle), np.sin(angle)])
    # Second ring (12 points) - but we only need 16 total, so 9 more
    # Actually let's do a proper hexagonal arrangement for 16 points
    # Use a pattern that approximates optimal packing
    # Let's use a slightly distorted grid or a known good configuration

    # 3. Random starts with different seeds
    for seed in range(20):
        np.random.seed(seed)
        pts = np.random.randn(n, d)
        candidates.append(normalize_points(pts))

    # 4. Circle-based configurations
    # Points on a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(normalize_points(circle))

    # 5. Two concentric circles
    inner_r = 0.5
    outer_r = 1.0
    angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    two_circ = np.vstack([
        inner_r * np.column_stack((np.cos(angles_inner), np.sin(angles_inner))),
        outer_r * np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    ])
    candidates.append(normalize_points(two_circ))

    # Evaluate initial candidates
    best_score = -1
    best_pts = None
    for pts in candidates:
        score = compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    # Local optimization using Nelder-Mead on the best candidate
    # We'll optimize a smooth surrogate: maximize log(d_min) - log(d_max)
    # or use a soft-min/soft-max approach

    def objective_flat(flat_pts):
        pts = flat_pts.reshape(n, d)
        dists = pdist(pts)
        # Use soft-min and soft-max for smoothness
        # soft-min: -log-sum-exp(-dists / tau)
        # soft-max: log-sum-exp(dists / tau)
        tau = 0.1
        soft_min = -np.log(np.sum(np.exp(-dists / tau))) * tau
        soft_max = np.log(np.sum(np.exp(dists / tau))) * tau
        if soft_max <= 0:
            return 0.0
        return (soft_min / soft_max) ** 2

    # Try Nelder-Mead optimization from the best candidate
    from scipy.optimize import minimize

    best_flat = best_pts.ravel()
    result = minimize(
        objective_flat,
        best_flat,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
    )

    if time.time() - start_time < deadline:
        opt_pts = normalize_points(result.x.reshape(n, d))
        opt_score = compute_score(opt_pts)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts

    # Try multiple restarts with different initial points if time permits
    restarts_done = 0
    max_restarts = 10
    while time.time() - start_time < deadline - 30 and restarts_done < max_restarts:
        np.random.seed(1000 + restarts_done)
        pts = np.random.randn(n, d)
        pts = normalize_points(pts)

        flat_pts = pts.ravel()
        res = minimize(
            objective_flat,
            flat_pts,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
        )

        if time.time() - start_time < deadline - 10:
            opt_pts = normalize_points(res.x.reshape(n, d))
            opt_score = compute_score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts

        restarts_done += 1

    # Final polish: tighter optimization on the best point found
    if time.time() - start_time < deadline - 5 and best_pts is not None:
        flat_best = best_pts.ravel()
        res_final = minimize(
            objective_flat,
            flat_best,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        polished_pts = normalize_points(res_final.x.reshape(n, d))
        polished_score = compute_score(polished_pts)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts

    # Ensure valid output
    if best_pts is None:
        best_pts = np.zeros((n, d))

    return best_pts


# EVOLVE-BLOCK-END
