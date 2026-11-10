# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time
import itertools


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    n = 16
    d = 2
    deadline = time.time() + 620  # safety margin before 630s kill

    def score(points):
        if points.shape != (n, d):
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(flat):
        pts = flat.reshape(n, d)
        s = score(pts)
        return -s

    def is_valid(pts):
        if pts.shape != (n, d):
            return False
        if not np.all(np.isfinite(pts)):
            return False
        return True

    best_points = None
    best_score = -np.inf

    def update_best(pts):
        nonlocal best_points, best_score
        if is_valid(pts):
            s = score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()

    # Generate multiple initial configurations
    configs = []

    # 1. Square grid
    grid = np.array(list(itertools.product([-1.5, -0.5, 0.5, 1.5], repeat=2)), dtype=float)
    configs.append(grid)

    # 2. Concentric rings: 1 center, 5 inner, 10 outer
    center = np.array([0.0, 0.0])
    inner_r = 1.0
    outer_r = 2.5
    inner_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi / 10
    inner_pts = np.column_stack([inner_r * np.cos(inner_angles), inner_r * np.sin(inner_angles)])
    outer_pts = np.column_stack([outer_r * np.cos(outer_angles), outer_r * np.sin(outer_angles)])
    ring_config = np.vstack([center.reshape(1, -1), inner_pts, outer_pts])
    configs.append(ring_config)

    # 3. Two concentric squares
    sq1 = np.array(list(itertools.product([-0.8, 0.8], repeat=2)), dtype=float)
    sq2 = np.array(list(itertools.product([-1.8, 1.8], repeat=2)), dtype=float)
    # Rotate outer square by 45 degrees
    theta = np.pi / 4
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    sq2_rot = sq2 @ rot.T
    two_sq_config = np.vstack([sq1, sq2_rot])
    configs.append(two_sq_config)

    # 4. Random configurations
    np.random.seed(42)
    for _ in range(5):
        rand_pts = np.random.randn(n, d) * 2
        configs.append(rand_pts)

    # Optimize each configuration
    for i, init_pts in enumerate(configs):
        if time.time() > deadline:
            break

        init_flat = init_pts.flatten()
        
        # Use Nelder-Mead with careful parameters
        try:
            res = minimize(
                objective,
                init_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            opt_pts = res.x.reshape(n, d)
            update_best(opt_pts)
        except Exception:
            pass

        # Also try a second pass with different parameters if time permits
        if time.time() > deadline - 30:
            continue

        try:
            res2 = minimize(
                objective,
                res.x,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-7,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            opt_pts2 = res2.x.reshape(n, d)
            update_best(opt_pts2)
        except Exception:
            pass

    # If we still have time, do a few more random restarts
    while time.time() < deadline - 10:
        np.random.seed(int(time.time() * 1000) % 100000)
        rand_pts = np.random.randn(n, d) * 2
        init_flat = rand_pts.flatten()
        try:
            res = minimize(
                objective,
                init_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-5,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            opt_pts = res.x.reshape(n, d)
            update_best(opt_pts)
        except Exception:
            break

    if best_points is not None:
        return best_points

    # Fallback: return the grid
    return configs[0]


# EVOLVE-BLOCK-END
