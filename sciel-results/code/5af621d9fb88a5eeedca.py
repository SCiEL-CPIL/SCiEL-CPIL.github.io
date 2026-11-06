"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 132.0  # Safety margin before 150s limit

    n = 8
    iu = np.triu_indices(n, k=1)
    num_pairs = len(iu[0])
    theta = 2 * np.pi * np.arange(n) / n

    def log_score(pts):
        """log of score Delta/n^n; Delta = product over ordered pairs of distances."""
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        # Guard against zero distances
        d_ij_safe = np.maximum(d_ij, 1e-15)
        log_delta = 2.0 * np.sum(np.log(d_ij_safe))
        return log_delta - n * np.log(n)

    def diameter(pts):
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.max(dists)

    def obj_and_grad(x, W, T):
        """
        Objective = -log_score + W * max(0, soft_diam - 2)
        soft_diam = (1/T) * logsumexp(T * (d_ij - 2))  ~ smooth max of (d_ij - 2)
        Analytical gradient.
        """
        pts = x.reshape(n, 2)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        d_ij = dists[iu]
        d_ij_safe = np.maximum(d_ij, 1e-15)
        log_delta = 2.0 * np.sum(np.log(d_ij_safe))
        log_s = log_delta - n * np.log(n)

        # Gradient of log_score
        grad_log = np.zeros_like(pts)
        for idx in range(num_pairs):
            i, j = iu[0][idx], iu[1][idx]
            d = d_ij[idx]
            if d < 1e-15:
                continue
            diff = pts[i] - pts[j]
            grad_log[i] += 2.0 * diff / (d * d)
            grad_log[j] -= 2.0 * diff / (d * d)

        # Soft diameter (log domain) and its gradient
        a = T * (d_ij - 2.0)
        m = np.max(a)
        e = np.exp(a - m)
        s = np.sum(e)
        smax = (m + np.log(s)) / T          # = logsumexp(a)/T
        w = e / s                            # weights over pairs

        val = -log_s + W * max(smax, 0.0)
        grad = -grad_log
        if smax > 0.0:
            for idx in range(num_pairs):
                i, j = iu[0][idx], iu[1][idx]
                d = d_ij[idx]
                if d < 1e-15:
                    continue
                diff = pts[i] - pts[j]
                coef = W * w[idx] / d
                grad[i] += coef * diff
                grad[j] -= coef * diff
        return val, grad

    def generate_initial_points():
        inits = []
        octagon = np.column_stack([np.cos(theta), np.sin(theta)])
        inits.append(octagon.copy())
        rng = np.random.RandomState(42)
        for scale in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
            inits.append((octagon + rng.randn(n, 2) * scale).copy())
        for angle_offset in [0, np.pi / 4, np.pi / 3, np.pi/6]:
            c1 = np.column_stack([np.cos(theta[:4]), np.sin(theta[:4])]) * 0.7
            c2 = np.column_stack([np.cos(theta[4:]), np.sin(theta[4:])]) * 0.7 + np.array([0.5, 0.0])
            inits.append(np.vstack([c1, c2]).copy())
        for _ in range(8):
            angles = rng.uniform(0, 2 * np.pi, n)
            radii = rng.uniform(0.3, 1.0, n)
            inits.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).copy())
        inits.append(np.array([
            [1, 1], [1, -1], [-1, -1], [-1, 1],
            [0, 1], [1, 0], [0, -1], [-1, 0]
        ], dtype=float).copy())
        # Two concentric squares
        r1, r2 = 0.6, 1.0
        sq1 = np.column_stack([r1*np.cos(theta[:4]), r1*np.sin(theta[:4])])
        sq2 = np.column_stack([r2*np.cos(theta[4:]+np.pi/8), r2*np.sin(theta[4:]+np.pi/8)])
        inits.append(np.vstack([sq1, sq2]).copy())
        # Alternating radii
        radii_alt = np.array([1.0, 0.6, 1.0, 0.6, 1.0, 0.6, 1.0, 0.6])
        inits.append(np.column_stack([radii_alt*np.cos(theta), radii_alt*np.sin(theta)]).copy())
        return inits

    def scaled_log_score(pts):
        d = diameter(pts)
        if d <= 1e-12:
            return -np.inf
        return log_score(pts * (2.0 / d))

    def run_lbfgs(x0, W, T, maxiter, maxfun):
        if time.time() > deadline - 8:
            return None
        try:
            res = minimize(
                lambda v: obj_and_grad(v, W, T)[0],
                x0,
                jac=lambda v: obj_and_grad(v, W, T)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': maxfun},
            )
            return res.x.reshape(n, 2)
        except Exception:
            return None

    # --- Stage 1: evaluate diverse starts ---
    best_score = -np.inf
    best_points = None
    for init_pts in generate_initial_points():
        if time.time() > deadline - 15:
            break
        ls = scaled_log_score(init_pts)
        if ls > best_score:
            best_score = ls
            d = diameter(init_pts)
            best_points = (init_pts * (2.0 / d)).copy()

    # --- Stage 2: constrained optimization from best starts ---
    scored = []
    for init_pts in generate_initial_points()[:12]:
        ls = scaled_log_score(init_pts)
        if ls > -np.inf:
            d = diameter(init_pts)
            scored.append((ls, init_pts * (2.0 / d)))
    scored.sort(key=lambda t: -t[0])

    penalty_weights = [5.0, 20.0, 50.0, 100.0, 200.0]
    temps = [50.0, 100.0, 200.0]
    for _, sp in scored[:8]:
        for pw in penalty_weights[:4]:
            for smt in temps[:3]:
                if time.time() > deadline - 20:
                    break
                opt = run_lbfgs(sp.flatten(), pw, smt, 300, 4000)
                if opt is None:
                    continue
                d = diameter(opt)
                if d <= 1e-10:
                    continue
                opt = opt * (2.0 / d)
                ls = log_score(opt)
                if ls > best_score:
                    best_score = ls
                    best_points = opt.copy()

    # --- Stage 3: local-refinement restarts around the best point ---
    if best_points is not None and time.time() < deadline - 25:
        rng2 = np.random.RandomState(7)
        for _ in range(8):
            if time.time() > deadline - 15:
                break
            pert = best_points + rng2.randn(n, 2) * 0.05
            for pw in [50.0, 100.0]:
                for smt in [100.0, 200.0]:
                    if time.time() > deadline - 10:
                        break
                    opt = run_lbfgs(pert.flatten(), pw, smt, 400, 6000)
                    if opt is None:
                        continue
                    d = diameter(opt)
                    if d <= 1e-10:
                        continue
                    opt = opt * (2.0 / d)
                    ls = log_score(opt)
                    if ls > best_score:
                        best_score = ls
                        best_points = opt.copy()

    # --- Stage 4: final polish (tighter constraints) ---
    if best_points is not None and time.time() < deadline - 15:
        for pw in [100.0, 200.0, 500.0]:
            for smt in [200.0, 400.0]:
                if time.time() > deadline - 8:
                    break
                opt = run_lbfgs(best_points.flatten(), pw, smt, 600, 8000)
                if opt is None:
                    continue
                d = diameter(opt)
                if d <= 1e-10:
                    continue
                opt = opt * (2.0 / d)
                ls = log_score(opt)
                if ls > best_score:
                    best_score = ls
                    best_points = opt.copy()

    # --- Stage 5: Nelder-Mead polish ---
    if best_points is not None and time.time() < deadline - 10:
        x0 = best_points.flatten()
        try:
            res = minimize(
                lambda v: -log_score(v.reshape(n, 2)),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
            )
            opt = res.x.reshape(n, 2)
            d = diameter(opt)
            if d > 1e-10:
                opt = opt * (2.0 / d)
                ls = log_score(opt)
                if ls > best_score:
                    best_score = ls
                    best_points = opt.copy()
        except Exception:
            pass

    # --- Finalize ---
    if best_points is None:
        best_points = np.column_stack([np.cos(theta), np.sin(theta)])

    d = diameter(best_points)
    if d > 1e-10:
        best_points = best_points * (2.0 / d)

    # Ensure distinctness
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-10:
                best_points[j] += np.array([1e-8, 1e-8])

    max_coord = np.max(np.abs(best_points))
    if max_coord > 1e5:
        best_points = best_points * (1e5 / max_coord)
        d = diameter(best_points)
        if d > 1e-10:
            best_points = best_points * (2.0 / d)

    return [[float(p[0]), float(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
