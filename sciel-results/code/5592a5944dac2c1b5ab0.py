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

    # ---- Complex pair representation: 4 complex numbers z_k, points are z_k and conj(z_k) ----
    # This enforces conjugate symmetry directly and avoids root-finding instability.
    # x = [Re(z0), Im(z0), Re(z1), Im(z1), Re(z2), Im(z2), Re(z3), Im(z3)]
    def build_complex_pairs(x):
        pts = np.array([
            [x[0], x[1]], [x[0], -x[1]],
            [x[2], x[3]], [x[2], -x[3]],
            [x[4], x[5]], [x[4], -x[5]],
            [x[6], x[7]], [x[6], -x[7]]
        ])
        return pts

    def neg_logscore_cp(x):
        pts = build_complex_pairs(x)
        return neg_logscore(pts)

    # ---- Discriminant-based smooth surrogate ----
    # The product of squared distances = |discriminant| for monic polynomial
    # disc(p) = prod_{i<j} (z_i - z_j)^2
    # We use this as a smooth function of the coefficients to guide optimization
    def neg_log_discriminant(c):
        """Smooth objective based on polynomial discriminant (real coefficients)."""
        try:
            coeffs = np.array([1.0] + list(c))
            # Compute discriminant via resultant or direct formula
            # For degree 8, use the formula: disc = (-1)^(n(n-1)/2) / a_n * Res(p, p')
            n_deg = 8
            # p'(z) coefficients
            p_coeffs = coeffs  # z^8 + c1 z^7 + ... + c8
            # p'(z) = 8z^7 + 7c1 z^6 + ... + c7
            dp_coeffs = np.array([n_deg * p_coeffs[0],
                                  (n_deg-1) * p_coeffs[1],
                                  (n_deg-2) * p_coeffs[2],
                                  (n_deg-3) * p_coeffs[3],
                                  (n_deg-4) * p_coeffs[4],
                                  (n_deg-5) * p_coeffs[5],
                                  (n_deg-6) * p_coeffs[6],
                                  (n_deg-7) * p_coeffs[7]])
            # Resultant via Sylvester matrix (simplified: use np.poly and evaluate)
            # Actually, use the fact that disc = prod (z_i - z_j)^2 for roots z_i
            # For numerical stability, compute via eigenvalues of companion matrix
            # Companion matrix eigenvalues = roots
            comp_mat = np.zeros((n_deg, n_deg))
            for i in range(n_deg - 1):
                comp_mat[i, i+1] = 1.0
            comp_mat[n_deg-1, :] = -p_coeffs[1:n_deg+1][::-1]
            try:
                eigvals = np.linalg.eigvals(comp_mat)
            except Exception:
                return 1e10
            if not np.all(np.isfinite(eigvals)):
                return 1e10
            # Compute product of squared differences
            log_prod = 0.0
            for i in range(n_deg):
                for j in range(i+1, n_deg):
                    diff = eigvals[i] - eigvals[j]
                    if abs(diff) < 1e-12:
                        return 1e10
                    log_prod += 2.0 * np.log(abs(diff))
            # Scale: diameter = max |z_i - z_j|, we need to normalize
            max_dist = 0.0
            for i in range(n_deg):
                for j in range(i+1, n_deg):
                    d = abs(eigvals[i] - eigvals[j])
                    if d > max_dist:
                        max_dist = d
            if max_dist < 1e-12:
                return 1e10
            # log_score = 2 * (sum log(d_ij) - 28*log(D) + 28*log(2) - 8*log(8))
            # But disc = prod (z_i-z_j)^2, so log(disc) = 2*sum log|z_i-z_j|
            # We want: 2*(sum log(d_ij_scaled) - 8*log(8)) where d_ij_scaled = d_ij * 2/D
            # = 2*(sum log(d_ij) + 28*log(2) - 28*log(D) - 8*log(8))
            log_score = 2.0 * (0.5 * log_prod + 28.0 * np.log(2.0) - 28.0 * np.log(max_dist) - 8.0 * np.log(8.0))
            return -log_score
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

    # ---- Extract complex pair parameters from a point set ----
    def extract_cp_params(pts):
        centroid = np.mean(pts, axis=0)
        pts_c = pts - centroid
        # Sort by y descending, then x ascending to get consistent pairing
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

    # ---- Extract polynomial coefficients from a point set (for discriminant objective) ----
    def extract_poly_params(pts):
        centroid = np.mean(pts, axis=0)
        pts_c = pts - centroid
        z = pts_c[:, 0] + 1j * pts_c[:, 1]
        coeffs = np.poly(z)
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

    # ---- Phase 2: Complex pair optimization (replaces unstable poly roots) ----
    cp_params = extract_cp_params(best_pts)

    for trial in range(8):
        if time_left() < 12:
            break
        if trial == 0:
            x0 = cp_params.copy()
        elif trial == 1:
            x0 = cp_params.copy()
            x0[1::2] = -x0[1::2]  # flip imaginary parts
        elif trial == 2:
            x0 = cp_params * 0.85
        elif trial == 3:
            x0 = cp_params * 1.15
        elif trial == 4:
            x0 = cp_params + 0.04 * rng.randn(8)
        elif trial == 5:
            x0 = cp_params + 0.1 * rng.randn(8)
        elif trial == 6:
            # Try rotating the configuration
            theta = np.pi / 12
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            pts_rot = best_pts @ rot.T
            x0 = extract_cp_params(pts_rot)
        else:
            x0 = cp_params + 0.2 * rng.randn(8)
        try:
            res = minimize(neg_logscore_cp, x0, method='L-BFGS-B',
                           bounds=[(-10, 10)] * 8,
                           options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 80})
            if np.all(np.isfinite(res.x)):
                cand = build_complex_pairs(res.x)
                s2 = compute_score(cand)
                if s2 > best_score:
                    best_score = s2
                    best_pts = cand
        except Exception:
            pass

    # ---- Phase 2b: Discriminant-based optimization (algebraic structure) ----
    poly_params = extract_poly_params(best_pts)
    for trial in range(4):
        if time_left() < 10:
            break
        if trial == 0:
            c0 = poly_params.copy()
        elif trial == 1:
            c0 = poly_params * 0.9
        else:
            c0 = poly_params + 0.03 * rng.randn(8)
        try:
            res = minimize(neg_log_discriminant, c0, method='L-BFGS-B',
                           bounds=[(-10, 10)] * 8,
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 60})
            if np.all(np.isfinite(res.x)):
                # Convert back to points via roots
                coeffs = np.array([1.0] + list(res.x))
                roots = np.roots(coeffs)
                pts = np.column_stack([roots.real, roots.imag])
                pts = pts - np.mean(pts, axis=0)
                s2 = compute_score(pts)
                if s2 > best_score:
                    best_score = s2
                    best_pts = pts
        except Exception:
            pass

    if time_left() < 15:
        return best_pts.tolist()

    # ---- Phase 3: Conjugate-symmetric 8D optimization (same as before but with more starts) ----
    cs_params = extract_cs_params(best_pts)

    n_cs_starts = 10
    for i in range(n_cs_starts):
        if time_left() < 10:
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
            # Rotation by small angle
            theta = np.pi / 16
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            pts_rot = best_pts @ rot.T
            x0 = extract_cs_params(pts_rot)
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

    # ---- Phase 4: Diverse 8D starts (reduced to save time for polish) ----
    reg_cs = np.array([1.0, 0.0,
                       1.0/math.sqrt(2), 1.0/math.sqrt(2),
                       0.0, 1.0,
                       -1.0/math.sqrt(2), 1.0/math.sqrt(2)])

    starts_diverse = [reg_cs.copy()]
    for pert in [0.03, 0.06, 0.1]:
        starts_diverse.append(reg_cs + pert * rng.randn(8))

    for t_try in [0.3, 0.5, 0.7, 1.0]:
        for phi_try in [0.0, np.pi/8, np.pi/4, np.pi/3]:
            pts = build_two_squares(t_try, phi_try)
            starts_diverse.append(extract_cs_params(pts))

    starts_diverse.append(cs_params + 0.05 * rng.randn(8))
    starts_diverse.append(cs_params + 0.1 * rng.randn(8))

    for x0 in starts_diverse:
        if time_left() < 6:
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

    try:
        res = minimize(objective_16d, best_pts.flatten(), method='L-BFGS-B',
                       bounds=[(-10, 10)] * 16,
                       options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100})
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
                       options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 100})
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
    for trial in range(3):
        if time_left() < 3:
            break
        perturbed = best_pts + 0.005 * rng2.randn(n, 2)
        try:
            res = minimize(objective_16d, perturbed.flatten(), method='L-BFGS-B',
                           bounds=[(-10, 10)] * 16,
                           options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 80})
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

    # ---- Phase 8: Ultra-tight final 8D polish (both CS and complex-pair) ----
    cs_ultra = extract_cs_params(best_pts)
    try:
        res = minimize(neg_logscore_cs, cs_ultra, method='L-BFGS-B',
                       bounds=[(-10, 10)] * 8,
                       options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 200})
        if np.all(np.isfinite(res.x)):
            cand = build_conj_symmetric(res.x)
            s2 = compute_score(cand)
            if s2 > best_score:
                best_score = s2
                best_pts = cand
    except Exception:
        pass

    # Also try complex-pair ultra polish
    cp_ultra = extract_cp_params(best_pts)
    try:
        res = minimize(neg_logscore_cp, cp_ultra, method='L-BFGS-B',
                       bounds=[(-10, 10)] * 8,
                       options={'maxiter': 10000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 200})
        if np.all(np.isfinite(res.x)):
            cand = build_complex_pairs(res.x)
            s2 = compute_score(cand)
            if s2 > best_score:
                best_score = s2
                best_pts = cand
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
