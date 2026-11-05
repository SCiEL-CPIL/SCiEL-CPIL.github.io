import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n = 8
    nlogn = n * np.log(n)
    iu, ju = np.triu_indices(n, k=1)  # 28 unordered pairs

    def exact_log_score(P):
        """Exact log(Delta/n^n) for 8 points (diameter-scaled). Returns -inf if degenerate."""
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dsq = np.sum(diff * diff, axis=2)
        d = np.sqrt(dsq[iu, ju])
        if np.any(d < 1e-12):
            return -np.inf
        diam = np.max(d)
        if diam < 1e-12:
            return -np.inf
        s = d * (2.0 / diam)
        return np.sum(np.log(s)) - nlogn

    def smooth_obj(x, beta):
        """Smooth surrogate: -log(score) with soft-max diameter. Minimize this."""
        P = x.reshape(n, 2)
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dsq = np.sum(diff * diff, axis=2)
        d = np.sqrt(dsq[iu, ju])
        if np.any(d < 1e-12):
            return 1e10
        # soft-max for the diameter
        scaled = d / beta
        m = np.max(scaled)
        lse = m + np.log(np.sum(np.exp(scaled - m)))
        diam = beta * lse
        if diam < 1e-12:
            return 1e10
        s = d * (2.0 / diam)
        s = np.maximum(s, 1e-15)
        return nlogn - np.sum(np.log(s))

    # ---------- Baseline: regular octagon ----------
    ang = 2.0 * np.pi * np.arange(n) / n
    reg_oct = np.column_stack([np.cos(ang), np.sin(ang)])
    best_score = exact_log_score(reg_oct)   # ~0.0
    best_points = reg_oct.copy()

    def try_update(P):
        nonlocal best_score, best_points
        if P is None or np.any(np.isnan(P)) or np.any(np.isinf(P)):
            return
        d = np.sqrt(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 1e-8:
            return
        s = exact_log_score(P)
        if s > best_score:
            best_score = s
            best_points = P.copy()

    # ---------- Build diverse initializations ----------
    starts = [reg_oct.flatten().copy()]

    # Two-squares with varied relative rotation (strong structural prior)
    for rot in np.linspace(0.0, 0.49, 25):
        a1 = 2.0 * np.pi * np.arange(4) / 4
        a2 = 2.0 * np.pi * (np.arange(4) / 4 + rot)
        sq1 = np.column_stack([np.cos(a1), np.sin(a1)])
        sq2 = np.column_stack([np.cos(a2), np.sin(a2)])
        starts.append(np.vstack([sq1, sq2]).flatten())

    # Antipodal-pair initializations (4 angles + their opposites)
    rng = np.random.RandomState(7)
    for _ in range(25):
        base = np.sort(rng.uniform(0.0, np.pi, 4))
        allang = np.concatenate([base, base + np.pi])
        pts = np.column_stack([np.cos(allang), np.sin(allang)])
        starts.append(pts.flatten())

    # Perturbed circle
    for _ in range(25):
        ba = 2.0 * np.pi * np.arange(n) / n + rng.uniform(-0.25, 0.25, n)
        rd = 1.0 + rng.uniform(-0.25, 0.25, n)
        pts = np.column_stack([np.cos(ba) * rd, np.sin(ba) * rd])
        starts.append(pts.flatten())

    # Random points
    for _ in range(20):
        starts.append(rng.uniform(-1.5, 1.5, (n, 2)).flatten())

    # ---------- Phase 1: smooth L-BFGS-B from every start (low beta) ----------
    betas_coarse = [8.0, 40.0]
    for s in starts:
        if time.time() > deadline - 45:
            break
        for beta in betas_coarse:
            if time.time() > deadline - 45:
                break
            try:
                res = minimize(lambda x, b=beta: smooth_obj(x, b), s,
                               method='L-BFGS-B',
                               options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < 1e9:
                    try_update(res.x.reshape(n, 2))
            except Exception:
                pass

    # ---------- Phase 2: refine current best with increasing beta ----------
    if time.time() < deadline - 25:
        for beta in [100.0, 500.0, 2000.0, 10000.0, 1e6]:
            if time.time() > deadline - 15:
                break
            try:
                res = minimize(lambda x: smooth_obj(x, beta), best_points.flatten(),
                               method='L-BFGS-B',
                               options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.fun < 1e9:
                    try_update(res.x.reshape(n, 2))
            except Exception:
                pass

    # ---------- Phase 3: global differential evolution (smooth surrogate) ----------
    if time.time() < deadline - 35:
        bounds = [(-4.0, 4.0)] * (2 * n)
        try:
            de_res = differential_evolution(lambda x: smooth_obj(x, 60.0), bounds,
                                            seed=42, maxiter=120, popsize=20,
                                            tol=1e-8, polish=True, updateinterval=10)
            if de_res.fun < 1e9:
                try_update(de_res.x.reshape(n, 2))
        except Exception:
            pass

    # ---------- Phase 4: local DE around the best solution ----------
    if time.time() < deadline - 20:
        try:
            bnd = [(best_points[i, j] - 0.6, best_points[i, j] + 0.6)
                   for i in range(n) for j in range(2)]
            de2 = differential_evolution(lambda x: smooth_obj(x, 300.0), bnd,
                                         seed=11, maxiter=100, popsize=15,
                                         tol=1e-10, polish=True)
            if de2.fun < 1e9:
                try_update(de2.x.reshape(n, 2))
        except Exception:
            pass

    # ---------- Phase 5: final exact polish (very high beta ~ exact max) ----------
    if time.time() < deadline - 5:
        for beta in [1e5, 1e7]:
            if time.time() > deadline - 3:
                break
            try:
                res = minimize(lambda x: smooth_obj(x, beta), best_points.flatten(),
                               method='L-BFGS-B',
                               options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-12})
                if res.fun < 1e9:
                    try_update(res.x.reshape(n, 2))
            except Exception:
                pass

    # ---------- Final validation ----------
    d = np.sqrt(np.sum((best_points[:, None, :] - best_points[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 1e-10:
        best_points = reg_oct
    if np.max(np.abs(best_points)) > 1e6:
        best_points = reg_oct

    return [[float(best_points[i, 0]), float(best_points[i, 1])] for i in range(n)]
