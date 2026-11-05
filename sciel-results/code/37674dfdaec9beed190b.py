import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 143.0

    n = 8
    i_idx, j_idx = np.triu_indices(n, k=1)

    def exact_log_score(points):
        """
        Computes the log of the score for a set of points.
        Score = 16 * prod(d_ij) / D^28
        log_score = ln(16) + sum(ln(d_ij)) - 28 * ln(D)
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq[i_idx, j_idx])
        
        if np.any(dists < 1e-12):
            return -np.inf
            
        D = np.max(dists)
        if D < 1e-12:
            return -np.inf
            
        log_prod = np.sum(np.log(dists))
        return np.log(16.0) + log_prod - 28.0 * np.log(D)

    def try_update(points):
        nonlocal best_score, best_points
        if points is None or np.any(np.isnan(points)):
            return
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_d = np.sqrt(np.min(dists_sq))
        if min_d < 1e-8:
            return
        s = exact_log_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Regular octagon as baseline
    angles = 2 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = exact_log_score(reg_oct)
    best_points = reg_oct.copy()

    # Two-square symmetric parameterization:
    # Square 1: radius r1, angle offset a1, vertices at angles a1 + k*pi/2
    # Square 2: radius r2, angle offset a2, vertices at angles a2 + k*pi/2
    # The regular octagon has r1=r2=1, a1=0, a2=pi/4
    # By symmetry, we can fix a1=0 and search over (r1, r2, a2)
    # Actually, let's search over (r1, r2, delta) where delta = a2 - a1, and fix a1=0

    def two_square_points(params):
        """
        params = [r1, r2, delta]
        Returns 8 points as (8, 2) array.
        Square 1: radius r1, angles 0, pi/2, pi, 3pi/2
        Square 2: radius r2, angles delta, delta+pi/2, delta+pi, delta+3pi/2
        """
        r1, r2, delta = params
        a1 = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
        a2 = delta + np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
        sq1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
        sq2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
        return np.vstack([sq1, sq2])

    def two_square_log_score(params):
        """
        Log score for the two-square configuration.
        """
        if params[0] < 1e-6 or params[1] < 1e-6:
            return -np.inf
        pts = two_square_points(params)
        return exact_log_score(pts)

    def neg_two_square_log_score(params):
        """
        Negative log score for minimization.
        """
        if params[0] < 1e-6 or params[1] < 1e-6:
            return np.inf
        pts = two_square_points(params)
        s = exact_log_score(pts)
        if s == -np.inf:
            return np.inf
        return -s

    # Fine grid search over (r1, r2, delta)
    # Regular octagon: r1=1, r2=1, delta=pi/4
    # Search in a reasonable range around the octagon
    r1_vals = np.linspace(0.5, 1.5, 15)
    r2_vals = np.linspace(0.5, 1.5, 15)
    delta_vals = np.linspace(0.0, np.pi/2, 20)

    grid_best_score = -np.inf
    grid_best_params = np.array([1.0, 1.0, np.pi/4])
    grid_best_points = reg_oct.copy()

    for r1 in r1_vals:
        if time.time() > deadline - 60:
            break
        for r2 in r2_vals:
            for delta in delta_vals:
                params = np.array([r1, r2, delta])
                s = two_square_log_score(params)
                if s > grid_best_score:
                    grid_best_score = s
                    grid_best_params = params
                    grid_best_points = two_square_points(params)

    # Refine the best grid point with L-BFGS-B
    if time.time() < deadline - 30:
        try:
            res = minimize(
                neg_two_square_log_score,
                grid_best_params,
                method='L-BFGS-B',
                bounds=[(0.3, 2.0), (0.3, 2.0), (-0.5, 1.0)],
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < np.inf:
                candidate_params = res.x
                candidate_points = two_square_points(candidate_params)
                try_update(candidate_points)
        except Exception:
            pass

    # Also try multiple random starts in the two-square family
    np.random.seed(42)
    for _ in range(30):
        if time.time() > deadline - 20:
            break
        r1 = np.random.uniform(0.5, 1.5)
        r2 = np.random.uniform(0.5, 1.5)
        delta = np.random.uniform(-0.3, 0.8)
        params = np.array([r1, r2, delta])
        try:
            res = minimize(
                neg_two_square_log_score,
                params,
                method='L-BFGS-B',
                bounds=[(0.3, 2.0), (0.3, 2.0), (-0.5, 1.0)],
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < np.inf:
                candidate_params = res.x
                candidate_points = two_square_points(candidate_params)
                try_update(candidate_points)
        except Exception:
            pass

    # High-precision polish on the best candidate in the two-square family
    if time.time() < deadline - 10:
        # Find the best two-square params from our search
        best_ts_score = -np.inf
        best_ts_params = grid_best_params
        for r1 in r1_vals[::3]:
            for r2 in r2_vals[::3]:
                for delta in delta_vals[::2]:
                    params = np.array([r1, r2, delta])
                    s = two_square_log_score(params)
                    if s > best_ts_score:
                        best_ts_score = s
                        best_ts_params = params

        try:
            res = minimize(
                neg_two_square_log_score,
                best_ts_params,
                method='L-BFGS-B',
                bounds=[(0.3, 2.0), (0.3, 2.0), (-0.5, 1.0)],
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < np.inf:
                candidate_params = res.x
                candidate_points = two_square_points(candidate_params)
                s = exact_log_score(candidate_points)
                if s > best_score:
                    best_score = s
                    best_points = candidate_points
        except Exception:
            pass

    # Final validation
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_d = np.sqrt(np.min(dists_sq))
    
    if min_d < 1e-8:
        best_points = reg_oct

    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e6:
        best_points = best_points / max_coord * 1e5

    return [[float(p[0]), float(p[1])] for p in best_points]

if __name__ == "__main__":
    pts = construct()
    if pts:
        print("Constructed 8 points.")
        print(pts[0])
