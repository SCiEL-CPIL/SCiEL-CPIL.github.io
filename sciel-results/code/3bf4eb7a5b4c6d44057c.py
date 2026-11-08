import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 140.0

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score_and_min_tris(points):
        """
        Compute min triangle area / convex hull area, and identify which triangles achieve the minimum.
        Returns (score, min_tris_indices, all_areas)
        """
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0, [], np.zeros(n_tris)
        if area_hull < 1e-9:
            return 0.0, [], np.zeros(n_tris)

        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)

        # Find triangles that are within a small tolerance of the minimum
        tol = 1e-6 * max(min_area, 1e-10)
        min_tris = np.where(areas <= min_area + tol)[0]

        if min_area < 0:
            return 0.0, [], areas
        return min_area / area_hull, min_tris, areas

    def objective(points_flat, weights=None):
        """
        Objective: minimize -score. If weights provided, use weighted min of areas.
        """
        pts = points_flat.reshape(-1, 2)
        try:
            centered = pts - pts.mean(axis=0)
            hull = ConvexHull(centered)
            area_hull = hull.volume
        except Exception:
            return 1e10
        if area_hull < 1e-9:
            return 1e10

        x = centered[:, 0]
        y = centered[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross

        if weights is not None:
            # Weighted minimum: use log-sum-exp trick for smoothness
            # Actually, use a smooth approximation: -log-sum-exp(-areas)
            # But that's expensive. Use: min(areas * weights) / sum(weights)
            w = np.asarray(weights, dtype=float)
            if w.sum() < 1e-15:
                w = np.ones(n_tris)
            w = w / w.sum()
            # Use a soft-min: -log(sum(exp(-areas/tau))) * tau
            tau = 0.01
            exp_neg = np.exp(-areas / tau)
            soft_min = tau * np.log(exp_neg.sum())
            return soft_min / area_hull
        else:
            return -np.min(areas) / area_hull

    def objective_smooth(points_flat, tau=0.05):
        """
        Smooth objective using log-sum-exp for differentiability.
        Minimizes: soft_min(areas) / hull_area
        """
        pts = points_flat.reshape(-1, 2)
        try:
            centered = pts - pts.mean(axis=0)
            hull = ConvexHull(centered)
            area_hull = hull.volume
        except Exception:
            return 1e10
        if area_hull < 1e-9:
            return 1e10

        x = centered[:, 0]
        y = centered[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross

        # Soft minimum via log-sum-exp
        exp_neg = np.exp(-areas / tau)
        soft_min = tau * np.log(exp_neg.sum())
        return soft_min / area_hull

    def objective_maxmin_smooth(points_flat, tau=0.05):
        """
        Maximize soft-min of areas / hull_area.
        Returns negative for minimization.
        """
        return -objective_smooth(points_flat, tau)

    def refine_with_binding_triangles(points, max_iter=50):
        """
        Iteratively refine by focusing on the binding (minimum) triangles.
        At each step, identify the min-area triangles and do a local optimization
        that penalizes deviations from equal area among those triangles.
        """
        best = points.copy()
        best_score, _, _ = compute_score_and_min_tris(best)

        for it in range(max_iter):
            if time.time() > start_time + time_limit - 5:
                break

            score, min_tris, areas = compute_score_and_min_tris(best)
            if len(min_tris) == 0:
                break

            # Get the vertices involved in the binding triangles
            involved = set()
            for t in min_tris:
                involved.add(tri_i[t])
                involved.add(tri_j[t])
                involved.add(tri_k[t])

            # Build a sub-objective: minimize variance of areas among binding tris
            # while keeping them as large as possible
            binding_areas = areas[min_tris]
            mean_ba = np.mean(binding_areas)
            var_ba = np.var(binding_areas)

            # Do a small Nelder-Mead on the full point set with a combined objective
            def sub_obj(pf):
                p = pf.reshape(-1, 2)
                s, mt, ar = compute_score_and_min_tris(p)
                if s <= 0:
                    return 1e10
                # Penalize if any triangle area drops below current min
                min_ar = np.min(ar)
                penalty = 0.0
                if min_ar < best_score * 0.999:
                    penalty = (best_score * 0.999 - min_ar) * 100
                return penalty - s

            try:
                res = opt.minimize(
                    sub_obj,
                    best.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11}
                )
                cand = res.x.reshape(-1, 2)
                cand_score, _, _ = compute_score_and_min_tris(cand)
                if cand_score > best_score:
                    best = cand.copy()
                    best_score = cand_score
            except Exception:
                pass

        return best, best_score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    best_score = -float('inf')
    best_points = None

    # Phase 1: Multiple initializations + L-BFGS-B with smooth objective
    rng = np.random.RandomState(12345)

    initializations = []

    # 1. Regular polygon
    initializations.append(_regular_polygon(N, 1.0))

    # 2. Perturbed regular polygon
    for _ in range(3):
        pts = _regular_polygon(N, 1.0)
        noise = rng.normal(0, 0.05, size=(N, 2))
        initializations.append(pts + noise)

    # 3. Random convex-ish configurations
    for _ in range(5):
        theta = np.sort(rng.uniform(0, 2 * np.pi, N))
        r = rng.uniform(0.7, 1.0, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        initializations.append(pts)

    # 4. Slightly non-regular: alternate radii
    for amp in [0.05, 0.1, 0.15]:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = 1.0 + amp * np.sin(2 * angles)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        initializations.append(pts)

    for init_pts in initializations:
        if time.time() > start_time + time_limit - 20:
            break

        # Try smooth objective with L-BFGS-B
        try:
            res = opt.minimize(
                objective_maxmin_smooth,
                init_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            cand = res.x.reshape(-1, 2)
            cand_score, _, _ = compute_score_and_min_tris(cand)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand.copy()
        except Exception:
            pass

        # Also try with a smaller tau for sharper min
        try:
            def obj_small_tau(pf):
                return -objective_smooth(pf, tau=0.02)
            res = opt.minimize(
                obj_small_tau,
                init_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            cand = res.x.reshape(-1, 2)
            cand_score, _, _ = compute_score_and_min_tris(cand)
            if cand_score > best_score:
                best_score = cand_score
                best_points = cand.copy()
        except Exception:
            pass

    # Phase 2: Refine using binding triangle identification
    if best_points is not None:
        if time.time() < start_time + time_limit - 10:
            refined, refined_score = refine_with_binding_triangles(best_points, max_iter=30)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined.copy()

    # Phase 3: Final polish with Nelder-Mead on raw objective
    if best_points is not None:
        if time.time() < start_time + time_limit - 5:
            try:
                def raw_obj(pf):
                    s, _, _ = compute_score_and_min_tris(pf.reshape(-1, 2))
                    return -s
                res = opt.minimize(
                    raw_obj,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                cand_score, _, _ = compute_score_and_min_tris(cand)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand.copy()
            except Exception:
                pass

    # Phase 4: One more round of binding-triangle refinement
    if best_points is not None:
        if time.time() < start_time + time_limit - 3:
            refined, refined_score = refine_with_binding_triangles(best_points, max_iter=20)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined.copy()

    if best_points is None:
        best_points = _regular_polygon(N, 1.0)

    return best_points
