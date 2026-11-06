import time
import math
import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0

    def time_left():
        return deadline - time.time()

    iu = np.triu_indices(n, k=1)

    def compute_score(pts):
        pts = np.asarray(pts, dtype=float)
        if not np.all(np.isfinite(pts)):
            return 0.0
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        Delta = prod_unordered ** 2
        return Delta / (n ** n)

    def neg_logscore(pts):
        pts = np.asarray(pts, dtype=float)
        if not np.all(np.isfinite(pts)):
            return 1e10
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    def neg_logscore_smooth(pts):
        """Smooth surrogate using log-sum-exp for soft-max of distances."""
        pts = np.asarray(pts, dtype=float)
        if not np.all(np.isfinite(pts)):
            return 1e10
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        # Soft-max via log-sum-exp with temperature
        temp = 0.01
        d_safe = np.maximum(d, 1e-15)
        log_d = np.log(d_safe)
        max_log_d = np.max(log_d)
        log_sum_exp = max_log_d + np.log(np.sum(np.exp(log_d - max_log_d)))
        # D_smooth = exp(log_sum_exp) / 28 (approximate max)
        D_smooth = np.exp(log_sum_exp)
        # Penalize if any distance is too small
        min_d = np.min(d)
        if min_d < 1e-8:
            return 1e10
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * log_sum_exp - 8.0 * np.log(8.0))
        return -log_score

    # ---- Regular octagon baseline ----
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    s_reg = compute_score(reg_pts)

    best_score = s_reg
    best_pts = reg_pts.copy()

    # ---- Two-square family ----
    def build_two_squares(t, phi):
        angles_outer = np.array([0.0, np.pi/2, np.pi, 3.0*np.pi/2])
        angles_inner = angles_outer + phi
        pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        pts_inner = t * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        return np.vstack([pts_outer, pts_inner])

    def neg_logscore_ts(x):
        t, phi = x
        if t <= 1e-6 or t > 5.0:
            return 1e10
        pts = build_two_squares(t, phi)
        return neg_logscore(pts)

    # ---- Conjugate-symmetric representation: 4 complex numbers -> 8 points ----
    def build_conj_symmetric(x):
        a0, b0, a1, b1, a2, b2, a3, b3 = x
        pts = np.array([
            [a0, b0], [a0, -b0],
            [a1, b1], [a1, -b1],
            [a2, b2], [a2, -b2],
            [a3, b3], [a3, -b3]
        ])
        return pts

    def neg_logscore_cs(x):
        pts = build_conj_symmetric(x)
        return neg_logscore(pts)

    # ---- Polynomial coefficient representation ----
    # p(z) = z^8 + c1*z^7 + c2*z^6 + c3*z^5 + c4*z^4 + c5*z^3 + c6*z^2 + c7*z + c8
    # With conjugate symmetry (real coefficients), if z is a root so is conj(z).
    # We optimize the 8 real coefficients.
    def build_from_poly_coeffs(c):
        # c = [c1, c2, c3, c4, c5, c6, c7, c8]
        coeffs = np.array([1.0, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
        roots = np.roots(coeffs)
        # Filter: keep only roots with |Im| > threshold OR all if symmetric
        # Actually we want all 8 roots; if some are complex, they come in conjugate pairs
        # Take real and imaginary parts
        pts = np.column_stack([roots.real, roots.imag])
        # Center at origin
        pts = pts - np.mean(pts, axis=0)
        return pts

    def neg_logscore_poly(c):
        try:
            pts = build_from_poly_coeffs(c)
            if not np.all(np.isfinite(pts)):
                return 1e10
            # Check distinctness
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            d = dists[iu]
            if np.min(d) < 1e-8:
                return 1e10
            return neg_logscore(pts)
        except Exception:
            return 1e10

    # ---- Extract conjugate-symmetric params from a point set ----
    def extract_cs_params(pts):
        centroid = np.mean(pts, axis=0)
        pts_c = pts - centroid
        y_order = np.argsort(pts_c[:, 1])[::-1]
        upper = pts_c[y_order[:4]]
        x_order = np.argsort(upper[:, 0])
        upper_sorted = upper[x_order]
        params = np.array([
            upper_sorted[0, 0], upper_sorted[0, 1],
            upper_sorted[1, 0], upper_sorted[1, 1],
            upper_sorted[2, 0], upper_sorted[2, 1],
            upper_sorted[3, 0], upper_sorted[3, 1]
        ])
        return params

    # ---- Extract polynomial coefficients from a point set ----
    def extract_poly_params(pts):
        centroid = np.mean(pts, axis=0)
        pts_c = pts - centroid
        # Form complex numbers
        z = pts_c[:, 0] + 1j * pts_c[:, 1]
        # Build monic polynomial with these roots
        coeffs = np.poly(z)
        # coeffs[0] should be ~1, take coeffs[1:]
        c = np.real(coeffs[1:])
        return c

    # ---- Phase 1: Quick two-square optimization ----
    rng = np.random.RandomState(42)

    t_grid = np.linspace(0.15, 1.6, 20)
    phi_grid = np.linspace(0.0, np.pi/2, 15)
    coarse_best = -np.inf
    coarse_t = 0.5
    coarse_phi = np.pi/4

    for t0 in t_grid:
        for phi0 in phi_grid:
            pts = build_two_squares(t0, phi0)
            s = compute_score(pts)
            if s > coarse_best:
                coarse_best = s
                coarse_t = t0
                coarse_phi = phi0

    if coarse_best > best_score:
        best_score = coarse_best
        best_pts = build_two_squares(coarse_t, coarse_phi)

    if time_left() < 30:
        return best_pts.tolist()

    try:
        res = minimize(neg_logscore_ts, np.array([coarse_t, coarse_phi]), method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-15})
        if np.all(np.isfinite(res.x)):
            t2, phi2 = float(res.x[0]), float(res.x[1])
            if t2 > 1e-6:
                cand = build_two_squares(t2, phi2 % (np.pi/2))
                s2 = compute_score(cand)
                if s2 > best_score:
                    best_score = s2
                    best_pts = cand
    except Exception:
        pass

    # Try other local minima in two-square space
    for t_alt in [0.4, 0.6, 0.8, 1.0, 1.2]:
        for phi_alt in [0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]:
            if time_left() < 25:
                break
            try:
                res = minimize(neg_logscore_ts, np.array([t_alt, phi_alt]), method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-15})
                if np.all(np.isfinite(res.x)):
                    t2, phi2 = float(res.x[0]), float(res.x[1])
                    if t2 > 1e-6:
                        cand = build_two_squares(t2, phi2 % (np.pi/2))
                        s2 = compute_score(cand)
                        if s2 > best_score:
                            best_score = s2
                            best_pts = cand
            except Exception:
                pass

    if time_left() < 20:
        return best_pts.tolist()

    # ---- Phase 2: Polynomial coefficient optimization ----
    poly_params = extract_poly_params(best_pts)

    for trial in range(8):
        if time_left() < 15:
            break
        if trial == 0:
            c0 = poly_params.copy()
        elif trial == 1:
            c0 = poly_params * 0.9
        elif trial == 2:
            c0 = poly_params * 1.1
        elif trial == 3:
            c0 = poly_params * 0.95
        elif trial == 4:
            c0 = poly_params * 1.05
        else:
            c0 = poly_params + 0.03 * rng.randn(8)
        try:
            res = minimize(neg_logscore_poly, c0, method='L-BFGS-B',
                           bounds=[(-10, 10)] * 8,
                           options={'maxiter': 4000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 80})
            if np.all(np.isfinite(res.x)):
                cand = build_from_poly_coeffs(res.x)
                s2 = compute_score(cand)
                if s2 > best_score:
                    best_score = s2
                    best_pts = cand
        except Exception:
            pass

    if time_left() < 15:
        return best_pts.tolist()

    # ---- Phase 3: Conjugate-symmetric 8D optimization ----
    cs_params = extract_cs_params(best_pts)

    n_cs_starts = 16
    for i in range(n_cs_starts):
        if time_left() < 12:
            break
        if i == 0:
            x0 = cs_params.copy()
        elif i == 1:
            x0 = cs_params.copy()
            x0[1::2] = -x0[1::2]
        elif i == 2:
            x0 = cs_params * 0.9
        elif i == 3:
            x0 = cs_params * 1.1
        elif i == 4:
            x0 = cs_params * 0.95
        elif i == 5:
            x0 = cs_params * 1.05
        elif i == 6:
            x0 = cs_params + 0.04 * rng.randn(8)
        elif i == 7:
            x0 = cs_params + 0.08 * rng.randn(8)
        else:
            x0 = cs_params + 0.05 * rng.randn(8)
        try:
            res = minimize(neg_logscore_cs, x0, method='L-BFGS-B',
                           bounds=[(-10, 10)] * 8,
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 80})
            if np.all(np.isfinite(res.x)):
                cand = build_conj_symmetric(res.x)
                s2 = compute_score(cand)
                if s2 > best_score:
                    best_score = s2
                    best_pts = cand
        except Exception:
            pass

    if time_left() < 10:
        return best_pts.tolist()

    # ---- Phase 4: Diverse 8D starts ----
    reg_cs = np.array([1.0, 0.0,
                       1.0/math.sqrt(2), 1.0/math.sqrt(2),
                       0.0, 1.0,
                       -1.0/math.sqrt(2), 1.0/math.sqrt(2)])

    starts_diverse = [reg_cs.copy()]
    for pert in [0.03, 0.05, 0.08, 0.12, 0.2]:
        starts_diverse.append(reg_cs + pert * rng.randn(8))

    for t_try in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]:
        for phi_try in [0.0, np.pi/12, np.pi/8, np.pi/6, np.pi/4, np.pi/3, 2*np.pi/3]:
            pts = build_two_squares(t_try, phi_try)
            starts_diverse.append(extract_cs_params(pts))

    for pert_scale in [0.05, 0.1, 0.15]:
        starts_diverse.append(cs_params + pert_scale * rng.randn(8))

    for x0 in starts_diverse:
        if time_left() < 8:
            break
        try:
            res = minimize(neg_logscore_cs, x0, method='L-BFGS-B',
                           bounds=[(-10, 10)] * 8,
                           options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50})
            if np.all(np.isfinite(res.x)):
                cand = build_conj_symmetric(res.x)
                s2 = compute_score(cand)
                if s2 > best_score:
                    best_score = s2
                    best_pts = cand
        except Exception:
            pass

    if time_left() < 8:
        return best_pts.tolist()

    # ---- Phase 5: Full 16D polish from best candidate ----
    def objective_16d(x):
        pts = x.reshape(n, 2)
        return neg_logscore(pts)

    def objective_16d_smooth(x):
        pts = x.reshape(n, 2)
        return neg_logscore_smooth(pts)

    # Try both regular and smooth objectives
    for obj_func in [objective_16d, objective_16d_smooth]:
        if time_left() < 8:
            break
        try:
            res = minimize(obj_func, best_pts.flatten(), method='L-BFGS-B',
                           bounds=[(-10, 10)] * 16,
                           options={'maxiter': 6000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 120})
            if np.all(np.isfinite(res.x)):
                refined = res.x.reshape(n, 2)
                s_ref = compute_score(refined)
                if s_ref > best_score:
                    best_score = s_ref
                    best_pts = refined
        except Exception:
            pass

    if time_left() < 6:
        return best_pts.tolist()

    # ---- Phase 6: Final 8D polish with tighter tolerances ----
    cs_final = extract_cs_params(best_pts)
    try:
        res = minimize(neg_logscore_cs, cs_final, method='L-BFGS-B',
                       bounds=[(-10, 10)] * 8,
                       options={'maxiter': 8000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 150})
        if np.all(np.isfinite(res.x)):
            cand = build_conj_symmetric(res.x)
            s2 = compute_score(cand)
            if s2 > best_score:
                best_score = s2
                best_pts = cand
    except Exception:
        pass

    if time_left() < 5:
        return best_pts.tolist()

    # ---- Phase 7: 16D from perturbed best ----
    rng2 = np.random.RandomState(999)
    for trial in range(4):
        if time_left() < 3:
            break
        pert_scale = 0.003 if trial < 2 else 0.008
        perturbed = best_pts + pert_scale * rng2.randn(n, 2)
        try:
            res = minimize(objective_16d, perturbed.flatten(), method='L-BFGS-B',
                           bounds=[(-10, 10)] * 16,
                           options={'maxiter': 4000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100})
            if np.all(np.isfinite(res.x)):
                refined = res.x.reshape(n, 2)
                s_ref = compute_score(refined)
                if s_ref > best_score:
                    best_score = s_ref
                    best_pts = refined
        except Exception:
            pass

    if time_left() < 3:
        return best_pts.tolist()

    # ---- Phase 8: Ultra-tight final 8D polish ----
    cs_ultra = extract_cs_params(best_pts)
    try:
        res = minimize(neg_logscore_cs, cs_ultra, method='L-BFGS-B',
                       bounds=[(-10, 10)] * 8,
                       options={'maxiter': 12000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 250})
        if np.all(np.isfinite(res.x)):
            cand = build_conj_symmetric(res.x)
            s2 = compute_score(cand)
            if s2 > best_score:
                best_score = s2
                best_pts = cand
    except Exception:
        pass

    # ---- Phase 9: Final ultra-tight 16D polish ----
    if time_left() > 5:
        try:
            res = minimize(objective_16d, best_pts.flatten(), method='L-BFGS-B',
                           bounds=[(-10, 10)] * 16,
                           options={'maxiter': 8000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 200})
            if np.all(np.isfinite(res.x)):
                refined = res.x.reshape(n, 2)
                s_ref = compute_score(refined)
                if s_ref > best_score:
                    best_score = s_ref
                    best_pts = refined
        except Exception:
            pass

    # Final validity check
    best_pts = np.asarray(best_pts, dtype=float)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    if len(set(map(tuple, np.round(best_pts, 10)))) < 8:
        best_pts = reg_pts

    return best_pts.tolist()
