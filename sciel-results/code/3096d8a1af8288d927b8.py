import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    timeout = 140.0

    indices = list(itertools.combinations(range(N), 3))
    n_combos = len(indices)

    def check_validity(points):
        x = points[:, 0]
        y = points[:, 1]
        if np.any(y < -1e-9):
            return False
        if np.any(y > SQRT3 * x + 1e-9):
            return False
        if np.any(y > -SQRT3 * x + SQRT3 + 1e-9):
            return False
        return True

    def min_area(points):
        x = points[:, 0]
        y = points[:, 1]
        min_a = float('inf')
        for i, j, k in indices:
            area = 0.5 * abs(x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j]))
            if area < min_a:
                min_a = area
        return min_a

    def _random_valid_points(n):
        points = []
        for _ in range(n):
            while True:
                u, v = np.random.rand(2)
                if u + v < 1:
                    x = v * 1.0 + (1 - u - v) * 0.5
                    y = (1 - u - v) * H
                    points.append([x, y])
                    break
        return np.array(points)

    def _soft_min_area(params, T=0.005):
        """Soft minimum of all triangle areas using log-sum-exp."""
        pts = params.reshape(N, 2)
        x = pts[:, 0]
        y = pts[:, 1]

        # Containment penalties
        pen = 0.0
        if np.any(y < 0):
            pen += np.sum(np.maximum(0, -y) ** 2)
        if np.any(y > SQRT3 * x):
            pen += np.sum(np.maximum(0, y - SQRT3 * x) ** 2)
        if np.any(y > -SQRT3 * x + SQRT3):
            pen += np.sum(np.maximum(0, y - (-SQRT3 * x + SQRT3)) ** 2)

        if pen > 1e-8:
            return 1e10 + pen

        areas = np.empty(n_combos)
        for idx, (i, j, k) in enumerate(indices):
            areas[idx] = 0.5 * abs(x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j]))

        areas_clipped = np.maximum(areas, 1e-12)
        log_sum_exp = np.log(np.sum(np.exp(-areas_clipped / T)))
        soft_min = -T * log_sum_exp

        return -soft_min

    def _objective(params, T=0.005):
        return _soft_min_area(params, T)

    bounds = [(0.0, 1.0), (0.0, H)] * N

    # Initial guess: grid-like placement
    def _initial_guess_grid():
        pts = np.zeros((N, 2))
        # 1 center
        pts[0] = [0.5, H / 3]
        # 3 near vertices
        r1 = 0.1
        pts[1] = [0.1 + r1, 0.0 + r1 * SQRT3 / 2]
        pts[2] = [0.9 - r1, 0.0 + r1 * SQRT3 / 2]
        pts[3] = [0.5, H - r1 * SQRT3]
        # 3 midpoints
        r2 = 0.05
        pts[4] = [0.25 - r2, H / 4 + r2 * SQRT3 / 2]
        pts[5] = [0.75 + r2, H / 4 + r2 * SQRT3 / 2]
        pts[6] = [0.5, H * 0.75 + r2 * SQRT3 / 2]
        # 3 ring at 1/3 height
        r3 = 0.15
        for k in range(3):
            angle = np.pi / 6 + 2 * np.pi * k / 3
            pts[7 + k, 0] = 0.5 + r3 * np.cos(angle)
            pts[7 + k, 1] = H / 3 + r3 * np.sin(angle)
        return pts

    best_points = _initial_guess_grid()
    best_score = min_area(best_points)

    # L-BFGS-B optimization
    try:
        res = minimize(
            _objective,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 20000},
        )
        if res.fun < 1e9:
            new_pts = res.x.reshape(N, 2)
            if check_validity(new_pts):
                new_score = min_area(new_pts)
                if new_score > best_score:
                    best_points = new_pts
                    best_score = new_score
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > timeout - 20:
        if not check_validity(best_points):
            return _random_valid_points(N)
        return best_points

    # Nelder-Mead refinement
    def _nm_objective(params):
        pts = params.reshape(N, 2)
        pen = 0.0
        if np.any(pts[:, 1] < 0):
            pen += np.sum(np.maximum(0, -pts[:, 1]) ** 2)
        if np.any(pts[:, 1] > SQRT3 * pts[:, 0]):
            pen += np.sum(np.maximum(0, pts[:, 1] - SQRT3 * pts[:, 0]) ** 2)
        if np.any(pts[:, 1] > -SQRT3 * pts[:, 0] + SQRT3):
            pen += np.sum(np.maximum(0, pts[:, 1] - (-SQRT3 * pts[:, 0] + SQRT3)) ** 2)
        if pen > 1e-8:
            return 1e10 + pen
        return _soft_min_area(params, T=0.003)

    try:
        res_nm = minimize(
            _nm_objective,
            best_points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 50000},
        )
        if res_nm.fun < 1e9:
            new_pts = res_nm.x.reshape(N, 2)
            if check_validity(new_pts):
                new_score = min_area(new_pts)
                if new_score > best_score:
                    best_points = new_pts
                    best_score = new_score
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > timeout - 10:
        if not check_validity(best_points):
            return _random_valid_points(N)
        return best_points

    # Multi-start random restarts
    np.random.seed(42)
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() - start_time > timeout - 5:
            break
        pts = _random_valid_points(N)
        try:
            res_r = minimize(
                _objective,
                pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res_r.fun < 1e9:
                if check_validity(res_r.x.reshape(N, 2)):
                    score_r = min_area(res_r.x.reshape(N, 2))
                    if score_r > best_score:
                        best_points = res_r.x.reshape(N, 2)
                        best_score = score_r
        except Exception:
            pass

    if not check_validity(best_points):
        return _random_valid_points(N)

    return best_points
