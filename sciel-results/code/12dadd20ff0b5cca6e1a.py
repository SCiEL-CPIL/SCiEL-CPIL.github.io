# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon maximizing minimum vertex score.
    Combines Fourier radial perturbation (structured, near-convex) with
    edge-length + exterior-angle representation (guarantees convexity).
    Uses differential evolution for global search and Nelder-Mead for local refinement.
    """
    n = 20
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin before 630s kill

    def _time_ok(remaining=10.0):
        return time.time() < deadline - remaining

    def _regular_polygon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def _normalize(points):
        centroid = np.mean(points, axis=0)
        pts = points - centroid
        dists = np.linalg.norm(pts, axis=1)
        mean_d = np.mean(dists)
        if mean_d <= 0.01:
            return None
        return pts / mean_d

    def _is_strictly_convex(points, tol=1e-12):
        """Check strict convexity: all cross products same sign, nonzero."""
        p0 = points
        p1 = np.roll(points, -1, axis=0)
        p2 = np.roll(points, -2, axis=0)
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        if np.any(np.abs(cross) < tol):
            return False
        if np.any(cross > 0) and np.any(cross < 0):
            return False
        # No repeated points
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(points[i] - points[j]) < 1e-10:
                    return False
        return True

    def _vertex_score(v_idx, polygon):
        v = polygon[v_idx]
        dists = np.sqrt(np.sum((polygon - v) ** 2, axis=1))
        mask = np.ones(n, dtype=bool)
        mask[v_idx] = False
        d = np.sort(dists[mask])  # 19 distances

        if d[0] < 0.05:
            return 0.0

        diffs = np.diff(d)
        k = int(np.argmin(diffs))
        D = (d[k] + d[k + 1]) / 2.0
        if D < 0.1:
            return 0.0

        if k == 0:
            sel = d[0:4]
        elif k == 17:
            sel = d[15:19]
        else:
            sel = d[k - 1:k + 3]

        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.maximum(sel / D, D / sel)
        raw = np.mean(ratios)
        if raw <= 0:
            return 0.0
        return min(1.0, 1.0 / raw)

    def _min_vertex_score(polygon):
        scores = np.array([_vertex_score(i, polygon) for i in range(n)])
        return float(np.min(scores))

    def _soft_min(scores, beta=50.0):
        """Differentiable approximation of min."""
        m = np.min(scores)
        return m - np.log(np.mean(np.exp(-beta * (scores - m)))) / beta

    # ===== Fourier radial representation =====
    K = 5  # number of Fourier harmonics (10 params total)
    n_fourier = 2 * K

    def _theta_angles():
        return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    def _fourier_to_polygon(params):
        """Convert Fourier params to (n,2) polygon vertices."""
        angles = _theta_angles()
        r = np.ones(n)
        for k in range(1, K + 1):
            a = params[2 * (k - 1)]
            b = params[2 * (k - 1) + 1]
            r += a * np.cos(k * angles) + b * np.sin(k * angles)
        r = np.maximum(r, 0.3)  # ensure positive radii
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.stack([x, y], axis=1)

    def _fourier_objective(params):
        """Objective for Fourier params: minimize negative soft-min score."""
        polygon = _fourier_to_polygon(params)
        norm_pts = _normalize(polygon)
        if norm_pts is None:
            return 1000.0
        if not _is_strictly_convex(norm_pts):
            return 1000.0 + np.sum(params ** 2)
        scores = np.array([_vertex_score(i, norm_pts) for i in range(n)])
        return -_soft_min(scores, beta=50.0)

    def _fourier_hard_score(params):
        """True minimum vertex score for Fourier params."""
        polygon = _fourier_to_polygon(params)
        norm_pts = _normalize(polygon)
        if norm_pts is None:
            return 0.0
        if not _is_strictly_convex(norm_pts):
            return 0.0
        scores = np.array([_vertex_score(i, norm_pts) for i in range(n)])
        return float(np.min(scores))

    # ===== Edge-length + exterior-angle representation =====
    def _edges_angles_to_polygon(edges, angles):
        """Convert edge lengths and exterior angles to (n,2) polygon."""
        directions = np.zeros(n)
        for i in range(1, n):
            directions[i] = directions[i - 1] + angles[i - 1]
        ex = edges * np.cos(directions)
        ey = edges * np.sin(directions)
        xs = np.concatenate([[0.0], np.cumsum(ex)])
        ys = np.concatenate([[0.0], np.cumsum(ey)])
        polygon = np.column_stack([xs[:n], ys[:n]])
        return polygon

    def _closure_error(edges, angles):
        """Sum of edge vectors should be zero for a closed polygon."""
        directions = np.zeros(n)
        for i in range(1, n):
            directions[i] = directions[i - 1] + angles[i - 1]
        ex = edges * np.cos(directions)
        ey = edges * np.sin(directions)
        cx = np.sum(ex)
        cy = np.sum(ey)
        return cx * cx + cy * cy

    def _params_to_polygon(params):
        """params: first n are edges, next n are exterior angles."""
        edges = params[:n]
        angles = params[n:]
        edges = np.maximum(edges, 0.05)
        angles = np.maximum(angles, 0.05)
        angle_sum = np.sum(angles)
        if angle_sum > 0:
            angles = angles / angle_sum * 2 * np.pi
        else:
            angles = np.full(n, 2 * np.pi / n)
        polygon = _edges_angles_to_polygon(edges, angles)
        return polygon

    def _score_from_params(params):
        """Compute score from edge/angle params, with closure penalty."""
        polygon = _params_to_polygon(params)
        norm_pts = _normalize(polygon)
        if norm_pts is None:
            return 1000.0
        if not _is_strictly_convex(norm_pts):
            return 1000.0
        edges = params[:n]
        angles = params[n:]
        angles = np.maximum(angles, 0.05)
        angle_sum = np.sum(angles)
        if angle_sum > 0:
            angles = angles / angle_sum * 2 * np.pi
        closure = _closure_error(edges, angles)
        if closure > 1e-4:
            return 1000.0 + closure
        scores = np.array([_vertex_score(i, norm_pts) for i in range(n)])
        return -_soft_min(scores, beta=50.0)

    def _hard_min_score(params):
        """True minimum vertex score (for final evaluation)."""
        polygon = _params_to_polygon(params)
        norm_pts = _normalize(polygon)
        if norm_pts is None:
            return 0.0
        if not _is_strictly_convex(norm_pts):
            return 0.0
        edges = params[:n]
        angles = params[n:]
        angles = np.maximum(angles, 0.05)
        angle_sum = np.sum(angles)
        if angle_sum > 0:
            angles = angles / angle_sum * 2 * np.pi
        closure = _closure_error(edges, angles)
        if closure > 1e-3:
            return 0.0
        scores = np.array([_vertex_score(i, norm_pts) for i in range(n)])
        return float(np.min(scores))

    # Initialize with regular polygon
    reg_pts = _regular_polygon()
    reg_norm = _normalize(reg_pts)
    best_score = _min_vertex_score(reg_norm) if reg_norm is not None else 0.0
    best_polygon = reg_norm.copy() if reg_norm is not None else reg_pts

    # ===== Phase 1: Fourier DE search =====
    if _time_ok(120):
        try:
            bounds_fourier = [(-0.15, 0.15)] * n_fourier
            result_f = differential_evolution(
                _fourier_objective,
                bounds_fourier,
                maxiter=300,
                popsize=20,
                tol=1e-7,
                seed=42,
                polish=True,
                init='sobol'
            )
            if result_f.fun < 0:
                cand_params = result_f.x
                cand_score = _fourier_hard_score(cand_params)
                if cand_score > best_score:
                    best_score = cand_score
                    cand_poly = _fourier_to_polygon(cand_params)
                    cand_norm = _normalize(cand_poly)
                    if cand_norm is not None and _is_strictly_convex(cand_norm):
                        best_polygon = cand_norm
        except Exception:
            pass

    # ===== Phase 2: Edge/angle DE search =====
    if _time_ok(120):
        try:
            # Initialize from regular polygon edges/angles
            angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
            pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
            edges_init = np.zeros(n)
            ext_angles = np.zeros(n)
            for i in range(n):
                p0 = pts[i]
                p1 = pts[(i + 1) % n]
                p2 = pts[(i + 2) % n]
                e1 = p1 - p0
                e2 = p2 - p1
                edges_init[i] = np.linalg.norm(e1)
                cross = e1[0] * e2[1] - e1[1] * e2[0]
                dot = np.dot(e1, e2)
                ext_angles[i] = np.arctan2(cross, dot)
                if ext_angles[i] < 0:
                    ext_angles[i] += 2 * np.pi

            rng = np.random.RandomState(42)
            edges_pert = edges_init * (1.0 + 0.05 * rng.randn(n))
            angles_pert = ext_angles + 0.02 * rng.randn(n)
            angles_pert = np.maximum(angles_pert, 0.1)
            angles_pert = angles_pert / np.sum(angles_pert) * 2 * np.pi
            edges_pert = np.maximum(edges_pert, 0.1)
            init_params = np.concatenate([edges_pert, angles_pert])

            bounds_ea = [(0.05, 3.0)] * n + [(0.05, 1.0)] * n
            result_ea = differential_evolution(
                _score_from_params,
                bounds_ea,
                maxiter=200,
                popsize=20,
                tol=1e-7,
                seed=123,
                polish=True,
                init='latinhypercube'
            )
            if result_ea.fun < 0:
                cand_params = result_ea.x
                cand_score = _hard_min_score(cand_params)
                if cand_score > best_score:
                    best_score = cand_score
                    best_params = cand_params
                    cand_poly = _params_to_polygon(cand_params)
                    cand_norm = _normalize(cand_poly)
                    if cand_norm is not None and _is_strictly_convex(cand_norm):
                        best_polygon = cand_norm
        except Exception:
            pass

    # ===== Phase 3: Nelder-Mead refinement on best Fourier params =====
    if _time_ok(60):
        try:
            # Find best Fourier params via local refinement
            best_fourier_params = np.zeros(n_fourier)
            best_fourier_score = 0.0
            for seed_try in range(5):
                rng_local = np.random.RandomState(seed_try * 7 + 1)
                local_params = rng_local.uniform(-0.15, 0.15, n_fourier)
                res_local = minimize(
                    _fourier_objective,
                    local_params,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                local_score = _fourier_hard_score(res_local.x)
                if local_score > best_fourier_score:
                    best_fourier_score = local_score
                    best_fourier_params = res_local.x
            if best_fourier_score > best_score:
                best_score = best_fourier_score
                cand_poly = _fourier_to_polygon(best_fourier_params)
                cand_norm = _normalize(cand_poly)
                if cand_norm is not None and _is_strictly_convex(cand_norm):
                    best_polygon = cand_norm
        except Exception:
            pass

    # ===== Phase 4: Final fine-tuning =====
    if _time_ok(30):
        try:
            # Try to refine the best polygon using edge/angle representation
            # Convert best_polygon to edge/angle params for refinement
            if best_polygon is not None:
                p0 = best_polygon
                p1 = np.roll(p0, -1, axis=0)
                p2 = np.roll(p0, -2, axis=0)
                e1 = p1 - p0
                e2 = p2 - p1
                edges_cur = np.linalg.norm(e1, axis=1)
                ext_angles_cur = np.zeros(n)
                for i in range(n):
                    cross = e1[i, 0] * e2[i, 1] - e1[i, 1] * e2[i, 0]
                    dot = np.dot(e1[i], e2[i])
                    ext_angles_cur[i] = np.arctan2(cross, dot)
                    if ext_angles_cur[i] < 0:
                        ext_angles_cur[i] += 2 * np.pi
                init_ea = np.concatenate([edges_cur, ext_angles_cur])

                res_final = minimize(
                    _score_from_params,
                    init_ea,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                final_score = _hard_min_score(res_final.x)
                if final_score > best_score:
                    best_score = final_score
                    final_poly = _params_to_polygon(res_final.x)
                    final_norm = _normalize(final_poly)
                    if final_norm is not None and _is_strictly_convex(final_norm):
                        best_polygon = final_norm
        except Exception:
            pass

    # Final validation and return
    final_norm = _normalize(best_polygon)
    if final_norm is None or not _is_strictly_convex(final_norm):
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        final_norm = _normalize(np.column_stack([np.cos(angles), np.sin(angles)]))

    return final_norm

# EVOLVE-BLOCK-END
