import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 142.0  # Leave 8s margin before the 150s kill

    # Precompute triple index arrays for vectorized area computation
    _ti, _tj, _tk = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                _ti.append(i); _tj.append(j); _tk.append(k)
    tri_i = np.array(_ti, dtype=np.int32)
    tri_j = np.array(_tj, dtype=np.int32)
    tri_k = np.array(_tk, dtype=np.int32)

    def _hull_area(points):
        try:
            return ConvexHull(points).volume
        except Exception:
            return 0.0

    def compute_score(points):
        """Return (min_triangle_area / hull_area, hull_area)."""
        area_hull = _hull_area(points)
        if area_hull < 1e-9:
            return 0.0, 0.0
        x = points[:, 0]; y = points[:, 1]
        v1x = x[tri_j] - x[tri_i]; v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]; v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        return float(np.min(areas)) / area_hull, area_hull

    def _soft_min_score(points, beta=50.0, area_ref=1.0):
        """
        Smooth surrogate for the min-triangle-area / hull-area ratio.
        Returns a value to MINIMIZE:  -soft_ratio  +  penalty*(1 - area/area_ref)^2
        """
        area_hull = _hull_area(points)
        if area_hull < 1e-9:
            return 1e6
        x = points[:, 0]; y = points[:, 1]
        v1x = x[tri_j] - x[tri_i]; v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]; v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)

        neg = -beta * areas
        m = np.max(neg)
        if m < -1e6:
            return 1e6
        soft_min = -(m + np.log(np.sum(np.exp(neg - m)))) / beta
        soft_ratio = soft_min / area_hull

        # Hull-area normalization penalty
        pen = 0.5 * (1.0 - area_hull / area_ref) ** 2
        return -soft_ratio + pen

    def _soft_objective(points_flat, beta=50.0, area_ref=1.0):
        return _soft_min_score(points_flat.reshape(-1, 2), beta, area_ref)

    def _hard_score_neg(points_flat):
        s, _ = compute_score(points_flat.reshape(-1, 2))
        return -s

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), phase=0.0):
        if n == 1:
            return np.array([center, center])
        a = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
        return np.column_stack([center[0] + radius * np.cos(a),
                                center[1] + radius * np.sin(a)])

    def _ring_config(r1, n1, r2=0.0, n2=0, n3=0, phase1=0.0, phase2=0.0, r3=0.0, phase3=0.0):
        """n1 outer + n2 middle + n3 inner (or 1 center) = N points."""
        parts = []
        if n1 > 0:
            parts.append(_regular_polygon(n1, r1, phase=phase1))
        if n2 > 0:
            parts.append(_regular_polygon(n2, r2, phase=phase2))
        if n3 == 1:
            parts.append(np.array([[0.0, 0.0]]))
        elif n3 > 1:
            parts.append(_regular_polygon(n3, r3 if r3 > 0 else 0.25 * max(r2, 0.3), phase=phase3))
        if not parts:
            return _regular_polygon(N, 1.0)
        return np.vstack(parts)

    def _equi_area_newton(points, target_area, max_iter=50, tol=1e-10):
        """
        Damped Newton solver for equi-area constraint on the K smallest triangles.
        Forces the areas of the K active triangles to equal target_area.
        Returns refined points.
        """
        pts = points.copy().astype(np.float64)
        K = 15  # number of active triangles to constrain
        
        for it in range(max_iter):
            x = pts[:, 0]; y = pts[:, 1]
            v1x = x[tri_j] - x[tri_i]; v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]; v2y = y[tri_k] - y[tri_i]
            areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
            
            # Find the K smallest
            idx_sorted = np.argsort(areas)
            active_idx = idx_sorted[:K]
            active_areas = areas[active_idx]
            
            # Check convergence
            max_dev = np.max(np.abs(active_areas - target_area))
            if max_dev < tol:
                break
            
            # Build Jacobian: d(area_m) / d(point_p) for active triangles m, points p
            # area = 0.5 * |cross(v1, v2)| where v1 = Pj - Pi, v2 = Pk - Pi
            # For a specific triangle (i, j, k):
            # dA/dPi = -0.5 * cross(v1, v2) * (1, 1) ... actually let's derive properly
            # A = 0.5 * (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
            # dA/dxi = 0.5 * (-(yk - yi) + (yj - yi)) = 0.5 * (yi - yk + yj - yi) = 0.5 * (yj - yk)
            # dA/dyi = 0.5 * (-(xk - xi) + (xj - xi)) = 0.5 * (xi - xk + xj - xi) = 0.5 * (xj - xk)
            # dA/dxj = 0.5 * (yk - yi)
            # dA/dyj = 0.5 * (-(xk - xi)) = 0.5 * (xi - xk)
            # dA/dxk = 0.5 * (-(yj - yi)) = 0.5 * (yi - yj)
            # dA/dyk = 0.5 * (xj - xi)
            
            # We are minimizing sum (A_m - target)^2 for active m
            # Gradient: sum_m 2(A_m - target) * dA_m/dp
            # Hessian is complex, use Gauss-Newton or simple gradient descent with line search
            
            grad = np.zeros((N, 2))
            for m in range(K):
                idx = active_idx[m]
                i = tri_i[idx]; j = tri_j[idx]; k = tri_k[idx]
                A = areas[idx]
                diff = A - target_area
                
                # Derivatives of the signed area (without 0.5*abs, assuming orientation is fixed)
                # Actually, since we use abs, the sign flips. Let's use the signed area for gradient.
                # signed_area = 0.5 * (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
                # If signed_area < 0, we flip signs of derivatives.
                # Let's just compute the gradient of 0.5 * |cross|
                # It is easier to just use numerical gradient or careful derivation.
                
                # Let's use the standard formula for gradient of triangle area w.r.t vertices
                # dA/dPi = 0.5 * (Pj - Pk) rotated 90 deg
                # dA/dPj = 0.5 * (Pk - Pi) rotated 90 deg
                # dA/dPk = 0.5 * (Pi - Pj) rotated 90 deg
                
                # Rotation 90 deg: (x, y) -> (-y, x)
                # dA/dPi = 0.5 * (-(yj - yk), xj - xk)
                # dA/dPj = 0.5 * (-(yk - yi), xk - xi)
                # dA/dPk = 0.5 * (-(yi - yj), xi - xj)
                
                # But wait, A = 0.5 * |...|
                # If the triangle orientation is consistent, we can use these.
                # Let's just compute the gradient numerically for safety and simplicity in this context,
                # or use the analytic one with sign correction.
                
                # Analytic gradient of 0.5 * |cross(v1, v2)|
                # Let C = cross(v1, v2). A = 0.5 * |C|.
                # dA/dC = 0.5 * sign(C)
                # dC/dPi = cross(v2, v1) = -C ... wait.
                # v1 = Pj - Pi, v2 = Pk - Pi
                # C = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
                # dC/dxi = -(yk-yi) + (yj-yi) = yj - yk
                # dC/dyi = -(xk-xi) + (xj-xi) = xj - xk
                # dC/dxj = yk - yi
                # dC/dyj = -(xk - xi)
                # dC/dxk = -(yj - yi)
                # dC/dyk = xj - xi
                
                sign_C = 1.0 if C > 0 else -1.0 # Need C value
                C = (x[j]-x[i])*(y[k]-y[i]) - (y[j]-y[i])*(x[k]-x[i])
                if C < 0: sign_C = -1.0
                else: sign_C = 1.0
                
                factor = 0.5 * sign_C * diff
                
                # dA/dPi
                grad[i, 0] += factor * (y[j] - y[k])
                grad[i, 1] += factor * (x[j] - x[k])
                # dA/dPj
                grad[j, 0] += factor * (y[k] - y[i])
                grad[j, 1] += factor * (-(x[k] - x[i]))
                # dA/dPk
                grad[k, 0] += factor * (-(y[j] - y[i]))
                grad[k, 1] += factor * (x[j] - x[i])
            
            # Line search
            step = 0.01
            for ls in range(10):
                new_pts = pts - step * grad
                # Check if new points are valid (non-degenerate)
                try:
                    ha = _hull_area(new_pts)
                    if ha < 1e-9:
                        step *= 0.5
                        continue
                    x_new = new_pts[:, 0]; y_new = new_pts[:, 1]
                    v1x_new = x_new[tri_j] - x_new[tri_i]; v1y_new = y_new[tri_j] - y_new[tri_i]
                    v2x_new = x_new[tri_k] - x_new[tri_i]; v2y_new = y_new[tri_k] - y_new[tri_i]
                    areas_new = 0.5 * np.abs(v1x_new * v2y_new - v1y_new * v2x_new)
                    idx_sorted_new = np.argsort(areas_new)
                    active_new = areas_new[idx_sorted_new[:K]]
                    obj_new = np.sum((active_new - target_area)**2)
                    
                    # Compare with current obj
                    obj_cur = np.sum((active_areas - target_area)**2)
                    
                    if obj_new < obj_cur:
                        pts = new_pts
                        break
                    else:
                        step *= 0.5
                except Exception:
                    step *= 0.5
        
        return pts

    def _optimize_from(start_pts, budget_deadline):
        """Multi-stage refine: annealed soft-min L-BFGS-B, then Equi-Area Newton, then NM hard polish."""
        best = start_pts.copy()
        best_s, _ = compute_score(best)
        current = start_pts.copy()
        area_ref = max(_hull_area(best), 1e-6)

        # Stage 1: annealed soft-min with progressive beta
        betas = [8.0, 20.0, 50.0, 100.0, 200.0, 400.0]
        for beta in betas:
            if time.time() > budget_deadline:
                break
            try:
                res = opt.minimize(
                    lambda x: _soft_objective(x, beta, area_ref),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 80},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s; best = cand.copy()
                current = cand
            except Exception:
                pass

        # Stage 2: Equi-Area Newton refinement
        if time.time() < budget_deadline - 2:
            try:
                # Estimate target area as current min area
                s_cur, _ = compute_score(best)
                area_hull_cur = _hull_area(best)
                target_A = s_cur * area_hull_cur
                # Run Newton
                cand = _equi_area_newton(best, target_A, max_iter=30, tol=1e-8)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s; best = cand.copy()
            except Exception:
                pass

        # Stage 3: Nelder-Mead polish on the hard objective
        if time.time() < budget_deadline - 2:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-11, 'fatol': 1e-13},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s; best = cand.copy()
            except Exception:
                pass

        # Stage 4: Powell refinement if time remains
        if time.time() < budget_deadline - 1:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best.flatten(),
                    method='Powell',
                    options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-12},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s; best = cand.copy()
            except Exception:
                pass

        return best, best_s

    # ---- Build diverse starting configurations ----
    starts = []
    # Symmetric ring configs (strong geometry prior)
    starts.append(_ring_config(1.0, 12, n3=1))                 # 1 center + 12 outer
    starts.append(_ring_config(1.0, 6, 0.5, 6, 1))             # 1 center + 6 mid + 6 outer
    starts.append(_ring_config(1.0, 7, 0.45, 5, 1))            # 1 center + 5 mid + 7 outer
    starts.append(_ring_config(1.0, 8, 0.4, 4, 1))             # 1 center + 4 mid + 8 outer
    starts.append(_ring_config(1.0, 5, 0.45, 7, 1))            # 1 center + 7 mid + 5 outer
    starts.append(_ring_config(1.0, 13, 0.0, 0, 0))            # pure 13-gon
    starts.append(_ring_config(1.0, 12, n3=1, phase1=np.pi / 12))  # phase-shifted 12 outer
    starts.append(_ring_config(1.0, 9, 0.6, 3, 1))             # 1 center + 3 mid + 9 outer
    starts.append(_ring_config(1.0, 10, 0.5, 2, 1))            # 1 center + 2 mid + 10 outer
    starts.append(_ring_config(1.0, 11, 0.4, 1, 1))            # 1 center + 1 mid + 11 outer
    # More ring variations with different radii
    starts.append(_ring_config(1.0, 6, 0.65, 6, 1))            # wider mid ring
    starts.append(_ring_config(1.0, 6, 0.35, 6, 1))            # narrower mid ring
    starts.append(_ring_config(1.0, 7, 0.55, 5, 1))            # wider mid ring
    starts.append(_ring_config(1.0, 7, 0.35, 5, 1))            # narrower mid ring
    # k_hull / k_inner polygon splits
    for k_hull in [3, 4, 5, 6, 7, 8, 10]:
        k_inner = N - k_hull
        hull = _regular_polygon(k_hull, 1.0)
        if k_inner == 0:
            starts.append(hull)
        elif k_inner == 1:
            starts.append(np.vstack([hull, np.array([[0.0, 0.0]])]))
        else:
            inner = _regular_polygon(k_inner, 0.35, phase=np.pi / k_inner)
            starts.append(np.vstack([hull, inner]))

    # ---- Search: allocate a per-start budget ----
    best_points = None
    best_score = -float('inf')
    n_starts = len(starts)
    reserve_final = 15.0
    search_budget = time_limit - reserve_final
    per_start = search_budget / n_starts

    for idx, st in enumerate(starts):
        if time.time() > start_time + time_limit - 12:
            break
        # Give earlier (ring) starts a little more, later ones a little less
        remaining_starts = n_starts - idx
        deadline = min(start_time + time_limit - 12,
                       start_time + time_limit - 12 + (remaining_starts - 1) * per_start)
        deadline = max(deadline, start_time + 5.0)
        try:
            cand, s = _optimize_from(st, deadline)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            s, _ = compute_score(st)
            if s > best_score:
                best_score = s
                best_points = st.copy()

    # ---- Final global polish on the best found ----
    if best_points is not None and time.time() < start_time + time_limit - 6:
        area_ref = max(_hull_area(best_points), 1e-6)
        for beta in [300.0, 800.0]:
            if time.time() > start_time + time_limit - 6:
                break
            try:
                res = opt.minimize(
                    lambda x: _soft_objective(x, beta, area_ref),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 80},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s; best_points = cand.copy()
            except Exception:
                pass
        
        # Final Equi-Area Newton pass
        if time.time() < start_time + time_limit - 4:
            try:
                s_cur, _ = compute_score(best_points)
                area_hull_cur = _hull_area(best_points)
                target_A = s_cur * area_hull_cur
                cand = _equi_area_newton(best_points, target_A, max_iter=50, tol=1e-10)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s; best_points = cand.copy()
            except Exception:
                pass

        if time.time() < start_time + time_limit - 3:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'adaptive': True, 'xatol': 1e-13, 'fatol': 1e-15},
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_score:
                    best_score = s; best_points = cand.copy()
            except Exception:
                pass

    if best_points is None:
        best_points = _regular_polygon(N, 1.0)

    return best_points

# EVOLVE-BLOCK-END
