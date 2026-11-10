import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n = 8
    # Pre-compute upper-triangle pair indices for speed
    iu = np.triu_indices(8, k=1)
    idx_i = iu[0]
    idx_j = iu[1]

    def calculate_score(points):
        """
        points: 8x2 array
        Returns: log of the scaled product of distances.
        Score = Delta / 8^8, where Delta is product of ordered pair distances.
        Delta = (prod_{i<j} d_ij)^2.
        log(Score) = 2 * sum_{i<j} log(d_ij) - 8 * log(8).
        Scaling: d_scaled = d * (2 / D).
        log(d_scaled) = log(d) + log(2) - log(D).
        Sum_{i<j} log(d_scaled) = Sum log(d) + 28 * log(2) - 28 * log(D).
        log(Score) = 2 * [Sum log(d) + 28 * log(2) - 28 * log(D)] - 8 * log(8).
        """
        diff = points[idx_i, :] - points[idx_j, :]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        if np.any(dists < 1e-12):
            return -np.inf
        D = np.max(dists)
        if D < 1e-12:
            return -np.inf
        sum_log_d = np.sum(np.log(dists))
        log_scaled_sum = sum_log_d + 28.0 * np.log(2.0) - 28.0 * np.log(D)
        log_score = 2.0 * log_scaled_sum - 8.0 * np.log(8.0)
        return log_score

    def build_two_squares(R1, R2, theta, dx, dy):
        """Build 8 points from two squares with radii R1,R2, rotation theta, offset (dx,dy) for second square."""
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        p1 = np.column_stack([R1 * np.cos(angles), R1 * np.sin(angles)])
        p2 = np.column_stack([R2 * np.cos(angles + theta), R2 * np.sin(angles + theta)])
        p2 = p2 + np.array([dx, dy])
        return np.vstack([p1, p2])

    def objective_5d(x):
        if time.time() > deadline:
            return 1e10
        R1, R2, theta, dx, dy = x
        if R1 < 0.05 or R2 < 0.05:
            return 1e10
        pts = build_two_squares(R1, R2, theta, dx, dy)
        s = calculate_score(pts)
        if np.isinf(s) or np.isnan(s):
            return 1e10
        return -s

    def objective_16(y):
        if time.time() > deadline:
            return 1e10
        pts = y.reshape(8, 2)
        s = calculate_score(pts)
        if np.isinf(s) or np.isnan(s):
            return 1e10
        return -s

    def refine(x0, max_iter=300):
        """Run Powell then Nelder-Mead from x0, return best (score, points)."""
        if time.time() > deadline - 5.0:
            return -np.inf, x0.reshape(8, 2)

        best_s = -np.inf
        best_x = x0.copy()

        # Try Powell
        try:
            res_p = minimize(
                objective_16, x0, method='Powell',
                options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            s_p = -res_p.fun
            if np.isfinite(s_p) and s_p > -1e9:
                best_s = s_p
                best_x = res_p.x
        except Exception:
            pass

        # Try Nelder-Mead on the result of Powell
        if time.time() < deadline - 5.0:
            try:
                res_nm = minimize(
                    objective_16, best_x, method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                s_nm = -res_nm.fun
                if np.isfinite(s_nm) and s_nm > best_s:
                    best_s = s_nm
                    best_x = res_nm.x
            except Exception:
                pass

        return best_s, best_x.reshape(8, 2)

    best_points = None
    best_score = -np.inf

    # 1. Regular Octagon
    angles_oct = np.linspace(0, 2*np.pi, 8, endpoint=False)
    pts_oct = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])
    s_oct = calculate_score(pts_oct)
    if s_oct > best_score:
        best_score = s_oct
        best_points = pts_oct

    # 2. DE on 5D two-squares manifold
    bounds_5d = [
        (0.05, 3.0),   # R1
        (0.05, 3.0),   # R2
        (0.0, np.pi/4),# theta
        (-1.0, 1.0),   # dx
        (-1.0, 1.0)    # dy
    ]

    candidates = []

    try:
        res = differential_evolution(
            objective_5d,
            bounds_5d,
            maxiter=300,
            popsize=30,
            tol=1e-10,
            seed=42,
            init='sobol',
            workers=1,
            polish=False
        )
        if res.fun < 1e9:
            x_opt = res.x
            pts_cand = build_two_squares(*x_opt)
            s_cand = calculate_score(pts_cand)
            candidates.append((s_cand, pts_cand))
            if s_cand > best_score:
                best_score = s_cand
                best_points = pts_cand
    except Exception:
        pass

    # 3. Try multiple symmetric starting points with refine (Powell + Nelder-Mead)
    if time.time() < deadline - 20.0:
        start_pts_list = [pts_oct.copy()]
        
        # Add some hand-crafted two-squares starting points
        for R1, R2, theta, dx, dy in [
            (1.0, 0.7, np.pi/8, 0.0, 0.0),
            (1.0, 0.5, np.pi/4, 0.0, 0.0),
            (1.0, 0.8, np.pi/6, 0.0, 0.0),
            (1.0, 0.6, np.pi/8, 0.1, 0.0),
            (1.0, 0.9, np.pi/12, 0.0, 0.1),
            (1.0, 0.4, np.pi/4, 0.0, 0.0),
            (1.0, 1.0, np.pi/8, 0.0, 0.0),
            (1.0, 0.3, np.pi/4, 0.0, 0.0),
        ]:
            start_pts_list.append(build_two_squares(R1, R2, theta, dx, dy))

        # Also add the DE result if available
        if candidates:
            start_pts_list.append(candidates[0][1].copy())

        for sp in start_pts_list:
            if time.time() > deadline - 5.0:
                break
            s_ref, pts_ref = refine(sp.flatten())
            if s_ref > best_score:
                best_score = s_ref
                best_points = pts_ref.copy()
                candidates.append((s_ref, best_points.copy()))

    # 4. If we have good candidates, do a final refine on the best
    if candidates and time.time() < deadline - 3.0:
        candidates.sort(key=lambda t: -t[0])
        best_cand = candidates[0][1]
        s_final, pts_final = refine(best_cand.flatten(), max_iter=500)
        if s_final > best_score:
            best_score = s_final
            best_points = pts_final

    if best_points is None:
        angles_oct = np.linspace(0, 2*np.pi, 8, endpoint=False)
        best_points = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])
        
    # Verify distinctness
    diffs = best_points[:, np.newaxis, :] - best_points[:, :, np.newaxis]
    d = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 1e-6:
        # Fallback to octagon
        angles_oct = np.linspace(0, 2*np.pi, 8, endpoint=False)
        best_points = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])

    # Scale to diameter 2
    D = np.max(d)
    if D > 0:
        best_points = best_points * (2.0 / D)
        
    return [[float(p[0]), float(p[1])] for p in best_points]
