import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_score(pts):
        """
        Compute the actual score: Delta / n^n.
        Delta is product over ordered pairs of distances after scaling to diameter 2.
        """
        n = 8
        pts = np.asarray(pts)
        if not np.all(np.isfinite(pts)):
            return 0.0

        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]

        # Check for coincident points
        if np.min(d) < 1e-12:
            return 0.0

        D = np.max(d)
        if D < 1e-12:
            return 0.0

        # Scale distances to diameter 2
        log_d_sum = np.sum(np.log(d))
        log_prod_unordered_scaled = log_d_sum + 28.0 * math.log(2.0) - 28.0 * math.log(D)
        log_score = 2.0 * log_prod_unordered_scaled - 8.0 * math.log(8.0)

        return math.exp(log_score)

    def log_score_from_points(pts):
        """Compute log score from points for optimization."""
        n = 8
        pts = np.asarray(pts)
        if not np.all(np.isfinite(pts)):
            return -np.inf

        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]

        if np.min(d) < 1e-9:
            return -1e100

        D = np.max(d)
        if D < 1e-9:
            return -1e100

        log_d_sum = np.sum(np.log(d))
        log_score = 2.0 * (log_d_sum + 28.0 * math.log(2.0) - 28.0 * math.log(D)) - 8.0 * math.log(8.0)
        return log_score

    # === PHASE 1: Two concentric squares optimization ===
    # Points: (±s1, ±s1) and (±s2, ±s2) where s = r/√2
    # Parameterize by log_s1, log_s2

    def two_square_log_score(params):
        log_s1, log_s2 = params
        s1 = math.exp(log_s1)
        s2 = math.exp(log_s2)

        if s1 < 1e-10 or s2 < 1e-10 or abs(s1 - s2) < 1e-10:
            return -1e100

        pts = np.array([
            [s1, s1], [s1, -s1], [-s1, s1], [-s1, -s1],
            [s2, s2], [s2, -s2], [-s2, s2], [-s2, -s2]
        ])
        return log_score_from_points(pts)

    def two_square_objective(params):
        return -two_square_log_score(params)

    # Grid search over ratio s1/s2
    best_ts_val = np.inf
    best_ts_params = None

    for ratio in np.logspace(-2, 0, 80):  # s1/s2 from 0.01 to 1
        for s2 in np.logspace(-1, 1, 30):  # s2 from 0.1 to 10
            s1 = ratio * s2
            if s1 < 1e-6 or s2 < 1e-6 or abs(s1 - s2) < 1e-6:
                continue
            params = np.array([math.log(s1), math.log(s2)])
            val = two_square_objective(params)
            if val < best_ts_val:
                best_ts_val = val
                best_ts_params = params.copy()

    # Refine with local optimization
    if best_ts_params is not None:
        try:
            res_ts = minimize(
                two_square_objective,
                best_ts_params,
                method='L-BFGS-B',
                bounds=[(-5, 5), (-5, 5)],
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res_ts.fun < best_ts_val:
                best_ts_val = res_ts.fun
                best_ts_params = res_ts.x.copy()
        except Exception:
            pass

    # Try multiple random starting points
    np.random.seed(42)
    for _ in range(30):
        if time.time() > deadline - 80.0:
            break
        log_s1 = np.random.uniform(-3, 3)
        log_s2 = np.random.uniform(-3, 3)
        if abs(math.exp(log_s1) - math.exp(log_s2)) < 0.01:
            continue
        params = np.array([log_s1, log_s2])
        try:
            res_ts = minimize(
                two_square_objective,
                params,
                method='L-BFGS-B',
                bounds=[(-5, 5), (-5, 5)],
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res_ts.fun < best_ts_val:
                best_ts_val = res_ts.fun
                best_ts_params = res_ts.x.copy()
        except Exception:
            pass

    best_score = 0.0
    best_points = None

    def update_best(pts, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Evaluate best two-square
    if best_ts_params is not None and best_ts_val < 1e99:
        log_s1, log_s2 = best_ts_params
        s1 = math.exp(log_s1)
        s2 = math.exp(log_s2)
        ts_pts = np.array([
            [s1, s1], [s1, -s1], [-s1, s1], [-s1, -s1],
            [s2, s2], [s2, -s2], [-s2, s2], [-s2, -s2]
        ])
        ts_score = compute_score(ts_pts)
        update_best(ts_pts, ts_score)

    # === PHASE 2: General 16-parameter search (8 points, 2D each) ===
    def general_objective(x):
        """x is 16 values: [x1,y1,x2,y2,...,x8,y8]"""
        pts = x.reshape(8, 2)
        ls = log_score_from_points(pts)
        if ls < -1e99:
            return 1e100
        return -ls

    # Initialize from two-square solution
    if best_ts_params is not None and best_ts_val < 1e99:
        log_s1, log_s2 = best_ts_params
        s1 = math.exp(log_s1)
        s2 = math.exp(log_s2)
        init_pts = np.array([
            [s1, s1], [s1, -s1], [-s1, s1], [-s1, -s1],
            [s2, s2], [s2, -s2], [-s2, s2], [-s2, -s2]
        ])
    else:
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        init_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    init_x = init_pts.flatten()

    # Local refinement from two-square init
    if time.time() < deadline - 50.0:
        try:
            res_gen = minimize(
                general_objective,
                init_x,
                method='L-BFGS-B',
                bounds=[(-10, 10)] * 16,
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if np.all(np.isfinite(res_gen.x)):
                gen_pts = res_gen.x.reshape(8, 2)
                gen_score = compute_score(gen_pts)
                update_best(gen_pts, gen_score)
        except Exception:
            pass

    # Differential evolution for general search
    if time.time() < deadline - 40.0:
        try:
            bounds_gen = [(-5, 5)] * 16
            res_de = differential_evolution(
                general_objective,
                bounds_gen,
                seed=42,
                maxiter=80,
                popsize=15,
                tol=1e-6,
                polish=True
            )
            if np.all(np.isfinite(res_de.x)):
                de_pts = res_de.x.reshape(8, 2)
                de_score = compute_score(de_pts)
                update_best(de_pts, de_score)
        except Exception:
            pass

    # More local refinement from best
    if time.time() < deadline - 15.0 and best_points is not None:
        best_x = best_points.flatten()
        try:
            res_final = minimize(
                general_objective,
                best_x,
                method='L-BFGS-B',
                bounds=[(-10, 10)] * 16,
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.all(np.isfinite(res_final.x)):
                final_pts = res_final.x.reshape(8, 2)
                final_score = compute_score(final_pts)
                update_best(final_pts, final_score)
        except Exception:
            pass

    # Final polish on two-square if still competitive
    if time.time() < deadline - 5.0 and best_ts_params is not None and best_ts_val < 1e99:
        try:
            res_ts_final = minimize(
                two_square_objective,
                best_ts_params,
                method='L-BFGS-B',
                bounds=[(-5, 5), (-5, 5)],
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res_ts_final.fun < best_ts_val:
                log_s1, log_s2 = res_ts_final.x
                s1 = math.exp(log_s1)
                s2 = math.exp(log_s2)
                ts_pts_final = np.array([
                    [s1, s1], [s1, -s1], [-s1, s1], [-s1, -s1],
                    [s2, s2], [s2, -s2], [-s2, s2], [-s2, -s2]
                ])
                ts_score_final = compute_score(ts_pts_final)
                update_best(ts_pts_final, ts_score_final)
        except Exception:
            pass

    # Construct final points
    if best_points is not None:
        pts = best_points
    else:
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])

    # Verify distinctness
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(8, k=1)
    d = dists[iu]

    if np.min(d) < 1e-6 or not np.all(np.isfinite(pts)):
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])

    max_coord = np.max(np.abs(pts))
    if max_coord > 1e6:
        pts = pts / max_coord * 1e5

    return pts.tolist()
