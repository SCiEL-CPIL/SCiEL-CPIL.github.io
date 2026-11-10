import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n = 8
    LOG_N_LOG_N = n * np.log(float(n))  # 8*log(8)

    # ---- Core scoring (ordered pairs, scaled to diameter 2) ----
    def score_points(points):
        pts = np.asarray(points, dtype=np.float64)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        i, j = np.triu_indices(n, k=1)
        d2u = d2[i, j]
        if np.any(d2u < 1e-24):
            return 0.0
        max_d2 = np.max(d2u)
        if max_d2 < 1e-24:
            return 0.0
        log_prod_ordered = 2.0 * (0.5 * np.sum(np.log(d2u)))  # = sum log d over unordered, x2
        # scale s = 2 / sqrt(max_d2); each of 56 ordered distances scales by s
        log_delta = 56.0 * (np.log(2.0) - 0.5 * np.log(max_d2)) + log_prod_ordered
        return float(np.exp(log_delta - LOG_N_LOG_N))

    # ---- Objective to MINIMIZE = -log(score), using ordered-pair form ----
    def objective(x):
        pts = x.reshape(n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        i, j = np.triu_indices(n, k=1)
        d2u = d2[i, j]
        if np.any(d2u < 1e-24):
            return 1e10
        max_d2 = np.max(d2u)
        if max_d2 < 1e-24:
            return 1e10
        log_prod_ordered = 2.0 * (0.5 * np.sum(np.log(d2u)))
        log_score = 56.0 * (np.log(2.0) - 0.5 * np.log(max_d2)) + log_prod_ordered - LOG_N_LOG_N
        return -log_score

    # ---- 5-parameter symmetric manifold: two concentric rotated squares ----
    # p_k = c + r1 * R(a1 + k*pi/2),  k=0..3
    # p_{4+k} = c + r2 * R(a2 + k*pi/2), k=0..3
    def manifold_to_points(theta):
        cx, cy, r1, r2, delta = theta
        a1 = 0.0
        a2 = delta
        c = np.array([cx, cy])
        p1 = c + r1 * np.column_stack([np.cos(a1 + np.arange(4) * (np.pi/2)),
                                       np.sin(a1 + np.arange(4) * (np.pi/2))])
        p2 = c + r2 * np.column_stack([np.cos(a2 + np.arange(4) * (np.pi/2)),
                                       np.sin(a2 + np.arange(4) * (np.pi/2))])
        return np.vstack([p1, p2])

    def manifold_obj(theta):
        pts = manifold_to_points(theta)
        return objective(pts.flatten())

    best_pts = None
    best_score = -1.0

    def consider(pts):
        nonlocal best_pts, best_score
        if pts is None:
            return
        pts = np.asarray(pts, dtype=np.float64).reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return
        sc = score_points(pts)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()

    # Regular octagon as a baseline
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    oct_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    consider(oct_pts)

    # ============ PHASE 1: Global search on the 5-param manifold ============
    # Bounds: centers near origin, radii positive, relative rotation in [0, pi/2]
    bounds = [(-1.5, 1.5), (-1.5, 1.5), (0.2, 4.0), (0.2, 4.0), (0.0, np.pi/2)]

    # Seed DE with the regular octagon's manifold params (r1=r2=1, delta=pi/4, c=0)
    seed_theta = np.array([0.0, 0.0, 1.0, 1.0, np.pi/4])

    if time.time() < deadline - 40.0:
        try:
            de_res = differential_evolution(
                manifold_obj, bounds,
                seed=42, maxiter=80, tol=1e-8,
                popsize=14, init='sobol',
                polish=False,
                updating='immediate',
                seed_init=[seed_theta.tolist()],
            )
            if de_res is not None and np.all(np.isfinite(de_res.x)):
                consider(manifold_to_points(de_res.x))
        except Exception:
            pass

    # Also try a handful of hand-crafted manifold starts + local polish
    if time.time() < deadline - 30.0:
        rng = np.random.default_rng(7)
        manifold_starts = [seed_theta]
        for _ in range(12):
            manifold_starts.append(np.array([
                rng.uniform(-1, 1), rng.uniform(-1, 1),
                rng.uniform(0.3, 3.5), rng.uniform(0.3, 3.5),
                rng.uniform(0.0, np.pi/2),
            ]))
        for th in manifold_starts:
            if time.time() > deadline - 30.0:
                break
            try:
                r = minimize(manifold_obj, th, method='L-BFGS-B',
                             bounds=bounds, options={'maxiter': 300, 'ftol': 1e-16})
                if np.all(np.isfinite(r.x)):
                    consider(manifold_to_points(r.x))
            except Exception:
                pass

    # ============ PHASE 2: Release to full 16D, multi-stage polish ============
    if best_pts is not None:
        x0 = best_pts.flatten()
        cur = x0

        # Stage A: L-BFGS-B
        if time.time() < deadline - 25.0:
            try:
                r = minimize(objective, cur, method='L-BFGS-B',
                             options={'maxiter': 600, 'ftol': 1e-16, 'gtol': 1e-13})
                if np.all(np.isfinite(r.x)) and r.fun < objective(cur):
                    cur = r.x
            except Exception:
                pass

        # Stage B: Powell (derivative-free, robust)
        if time.time() < deadline - 18.0:
            try:
                r = minimize(objective, cur, method='Powell',
                             options={'maxiter': 300, 'xtol': 1e-13, 'ftol': 1e-13})
                if np.all(np.isfinite(r.x)) and r.fun < objective(cur):
                    cur = r.x
            except Exception:
                pass

        # Stage C: Nelder-Mead high-precision
        if time.time() < deadline - 12.0:
            try:
                r = minimize(objective, cur, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14})
                if np.all(np.isfinite(r.x)) and r.fun < objective(cur):
                    cur = r.x
            except Exception:
                pass

        consider(cur.reshape(n, 2))

        # ============ PHASE 3: Local basin-hopping around best, in 16D ============
        if time.time() < deadline - 8.0:
            rng2 = np.random.default_rng(123)
            base = cur
            for trial in range(40):
                if time.time() > deadline - 8.0:
                    break
                scale = 0.02 * (0.5 ** (trial // 15))
                pert = base + rng2.normal(0, scale, (n, 2))
                try:
                    r = minimize(objective, pert.flatten(), method='L-BFGS-B',
                                 options={'maxiter': 200, 'ftol': 1e-15})
                    if np.all(np.isfinite(r.x)) and r.fun < objective(base):
                        base = r.x
                        consider(base.reshape(n, 2))
                except Exception:
                    pass

    # ============ PHASE 4: Final ultra-precise polish ============
    if best_pts is not None and time.time() < deadline - 3.0:
        try:
            r = minimize(objective, best_pts.flatten(), method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-15})
            if np.all(np.isfinite(r.x)) and r.fun < objective(best_pts.flatten()):
                best_pts = r.x.reshape(n, 2)
        except Exception:
            pass

    # Safety: ensure distinct, finite, bounded
    if best_pts is None:
        best_pts = oct_pts
    best_pts = np.asarray(best_pts, dtype=np.float64)
    # clamp coordinates well within bound
    best_pts = np.clip(best_pts, -1e5, 1e5)

    return [[float(p[0]), float(p[1])] for p in best_pts]
