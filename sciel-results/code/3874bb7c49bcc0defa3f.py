# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 630s kill

    def exact_score(pts):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            pts = pts.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def exact_score_flat(x):
        return exact_score(x.reshape(n, d))

    def exact_objective(x):
        return -exact_score_flat(x)

    best_x = None
    best_score = 0.0

    def update_best(x, score):
        nonlocal best_x, best_score
        if score > best_score:
            best_score = score
            best_x = x.copy()

    # Phase 0: Try known good configurations as starting points
    # 4x4 grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    # Center it
    grid -= grid.mean(axis=0)
    # Normalize
    dmax = np.max(pdist(grid))
    if dmax > 0:
        grid /= dmax
    update_best(grid.ravel(), exact_score(grid))

    # Hexagonal lattice: 4 rows, 4 cols with offset
    hex_pts = []
    for i in range(4):
        for j in range(4):
            x = j + 0.5 * (i % 2)
            y = i * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts -= hex_pts.mean(axis=0)
    dmax = np.max(pdist(hex_pts))
    if dmax > 0:
        hex_pts /= dmax
    update_best(hex_pts.ravel(), exact_score(hex_pts))

    # Ring configuration: 1 center + 15 on circle
    ring = np.zeros((16, 2))
    angles = np.linspace(0, 2 * np.pi, 15, endpoint=False)
    ring[1:, 0] = np.cos(angles)
    ring[1:, 1] = np.sin(angles)
    update_best(ring.ravel(), exact_score(ring))

    # Two rings: 8 inner + 8 outer
    two_ring = np.zeros((16, 2))
    angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    two_ring[:8, 0] = 0.5 * np.cos(angles1)
    two_ring[:8, 1] = 0.5 * np.sin(angles1)
    two_ring[8:, 0] = np.cos(angles2)
    two_ring[8:, 1] = np.sin(angles2)
    update_best(two_ring.ravel(), exact_score(two_ring))

    # Phase 1: Differential Evolution (global search)
    if time.time() < deadline - 120:
        try:
            res_de = differential_evolution(
                exact_objective,
                [(-1, 1)] * (n * d),
                maxiter=300,
                popsize=25,
                seed=42,
                tol=1e-6,
                init='latinhypercube',
                polish=False,
                update_interval=1
            )
            if time.time() < deadline - 10:
                s = -res_de.fun
                if s > best_score:
                    update_best(res_de.x, s)
        except Exception:
            pass

    # Phase 2: Nelder-Mead local refinement on exact objective (handles non-smoothness)
    if best_x is not None and time.time() < deadline - 60:
        try:
            res_nm = minimize(
                exact_objective,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if time.time() < deadline - 10:
                s = -res_nm.fun
                if s > best_score:
                    update_best(res_nm.x, s)
        except Exception:
            pass

    # Phase 3: Multi-start Nelder-Mead from diverse initializations
    if time.time() < deadline - 40:
        # Generate diverse starting points
        starts = []

        # Random starts
        np.random.seed(123)
        for _ in range(10):
            x = np.random.uniform(-1, 1, (n * d))
            starts.append(x)

        # Circle with noise
        for noise_scale in [0.05, 0.1, 0.2]:
            x = np.zeros(n * d)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            x[0::2] = np.cos(angles) + np.random.normal(0, noise_scale, n)
            x[1::2] = np.sin(angles) + np.random.normal(0, noise_scale, n)
            starts.append(x)

        # Grid with noise
        for noise_scale in [0.05, 0.1]:
            x = np.zeros(n * d)
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    x[idx * 2] = i + np.random.normal(0, noise_scale)
                    x[idx * 2 + 1] = j + np.random.normal(0, noise_scale)
            starts.append(x)

        # Hex lattice with noise
        for noise_scale in [0.05, 0.1]:
            x = np.zeros(n * d)
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    px = j + 0.5 * (i % 2)
                    py = i * np.sqrt(3) / 2
                    x[idx * 2] = px + np.random.normal(0, noise_scale)
                    x[idx * 2 + 1] = py + np.random.normal(0, noise_scale)
            starts.append(x)

        for x_start in starts:
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(
                    exact_objective,
                    x_start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-7,
                        'fatol': 1e-9,
                        'adaptive': True
                    }
                )
                if time.time() > deadline - 10:
                    break
                s = -res.fun
                if s > best_score:
                    update_best(res.x, s)
            except Exception:
                continue

    # Phase 4: Final polish with L-BFGS-B using soft-min approximation
    if best_x is not None and time.time() < deadline - 10:
        def soft_min(dists, t=0.05):
            return -np.log(np.sum(np.exp(-dists / t)) / len(dists)) * t

        def smooth_objective(x):
            pts = x.reshape(n, d)
            dists = pdist(pts)
            d_max = np.max(dists)
            if d_max < 1e-12:
                return 0.0
            dists_norm = dists / d_max
            d_min_norm = soft_min(dists_norm)
            return -d_min_norm

        try:
            res_lbfgs = minimize(
                smooth_objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            if time.time() < deadline - 5:
                s = exact_score_flat(res_lbfgs.x)
                if s > best_score:
                    update_best(res_lbfgs.x, s)
        except Exception:
            pass

    # Final Nelder-Mead polish
    if best_x is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(
                exact_objective,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True
                }
            )
            s = -res_final.fun
            if s > best_score:
                update_best(res_final.x, s)
        except Exception:
            pass

    if best_x is None:
        # Fallback: 4x4 grid
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        grid -= grid.mean(axis=0)
        dmax = np.max(pdist(grid))
        if dmax > 0:
            grid /= dmax
        best_x = grid.ravel()

    return best_x.reshape(n, d)

# EVOLVE-BLOCK-END
