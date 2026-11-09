# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _normalize(polygon):
    """Subtract centroid and divide by mean distance to centroid."""
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists = np.linalg.norm(centered, axis=1)
    mean_dist = dists.mean()
    if mean_dist <= 0.01:
        return centered
    return centered / mean_dist


def _vertex_score(distances):
    """Compute the per-vertex score from sorted distances to other vertices."""
    d = np.sort(distances)
    if np.any(d < 0.05):
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        idx = [0, 1, 2, 3]
    elif k == 17:
        idx = [15, 16, 17, 18]
    else:
        idx = [k - 1, k, k + 1, k + 2]
    four_d = d[idx]
    raw = np.mean(np.maximum(four_d / D, D / four_d))
    return min(1.0, 1.0 / raw)


def _polygon_score(polygon):
    """Compute the overall score: min vertex score over all vertices."""
    n = len(polygon)
    scores = []
    for i in range(n):
        v = polygon[i]
        dists = np.sqrt(np.sum((polygon - v) ** 2, axis=1))
        dists = np.delete(dists, i)
        scores.append(_vertex_score(dists))
    return min(scores)


def _is_valid_convex(polygon):
    """Check strictly convex: no repeated points, all turns same sign nonzero, total turning +-2*pi."""
    n = len(polygon)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(polygon[i] - polygon[j]) < 1e-10:
                return False
    cross_prods = []
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        cross_prods.append(cross)
    cross_prods = np.array(cross_prods)
    if np.any(np.abs(cross_prods) < 1e-12):
        return False
    if np.any(cross_prods > 0) and np.any(cross_prods < 0):
        return False
    total_turn = 0.0
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        e1 = p1 - p0
        e2 = p2 - p1
        angle1 = np.arctan2(e1[1], e1[0])
        angle2 = np.arctan2(e2[1], e2[0])
        d_angle = angle2 - angle1
        while d_angle > np.pi:
            d_angle -= 2 * np.pi
        while d_angle < -np.pi:
            d_angle += 2 * np.pi
        total_turn += d_angle
    if abs(abs(total_turn) - 2 * np.pi) > 1e-6:
        return False
    return True


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices that maximizes the minimum
    per-vertex equidistance score.
    """
    n = 20
    t0 = time.time()
    deadline = t0 + 590  # Safety margin before 630s limit

    def _elapsed():
        return time.time() - t0

    # Start from regular polygon
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    best_polygon = _normalize(best_polygon)
    best_score = _polygon_score(best_polygon)

    def _get_polygon(params):
        a = np.sort(params % (2 * np.pi))
        x = np.cos(a)
        y = np.sin(a)
        poly = np.stack([x, y], axis=1)
        return poly

    def _all_vertex_scores(polygon):
        """Return array of per-vertex scores."""
        n = len(polygon)
        scores = np.zeros(n)
        for i in range(n):
            v = polygon[i]
            dists = np.sqrt(np.sum((polygon - v) ** 2, axis=1))
            dists = np.delete(dists, i)
            scores[i] = _vertex_score(dists)
        return scores

    def _epigraph_objective(params, t):
        """
        Epigraph formulation: minimize -t + penalty for vertices below t.
        This directly pushes t up while keeping all vertex scores >= t.
        """
        poly = _get_polygon(params)
        if not _is_valid_convex(poly):
            return 10.0  # Large penalty
        scores = _all_vertex_scores(poly)
        # Penalty: sum of max(0, t - s_i)^2 for smoothness
        violations = np.maximum(0.0, t - scores)
        penalty = np.sum(violations ** 2)
        return -t + 10.0 * penalty

    def _softmin_objective(params, t):
        """
        Soft-min surrogate: -softmin(scores) where softmin approximates min.
        """
        poly = _get_polygon(params)
        if not _is_valid_convex(poly):
            return 10.0
        scores = _all_vertex_scores(poly)
        beta = 80.0
        min_s = np.min(scores)
        shifted = scores - min_s
        soft_min = min_s - np.log(np.sum(np.exp(-beta * shifted))) / beta
        return -soft_min

    def _neg_softmin_score(params):
        return _softmin_objective(params, 0.0)

    def _neg_epigraph(params):
        # Use current best score as initial t
        return _epigraph_objective(params, best_score)

    # Phase 1: Nelder-Mead with multiple restarts (broad exploration)
    rng = np.random.default_rng(42)
    x0 = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    n_restarts = 10
    for r in range(n_restarts):
        if _elapsed() > deadline - 150:
            break
        if r == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + rng.normal(0, 0.06, n)

        try:
            result = minimize(_neg_softmin_score, x_start, method='Nelder-Mead',
                            options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < 0:
                a = np.sort(result.x % (2 * np.pi))
                poly = np.stack([np.cos(a), np.sin(a)], axis=1)
                if _is_valid_convex(poly):
                    exact_s = _polygon_score(poly)
                    if exact_s > best_score:
                        best_score = exact_s
                        best_polygon = poly
        except Exception:
            pass

    # Phase 2: Epigraph refinement - directly optimize t with vertex constraints
    if _elapsed() < deadline - 100:
        a_best = np.arctan2(best_polygon[:, 1], best_polygon[:, 0])
        a_best = np.sort(a_best % (2 * np.pi))

        # Use SLSQP with epigraph: variables are [angles..., t]
        def _epigraph_slsqp(x):
            params = x[:-1]
            t = x[-1]
            return _epigraph_objective(params, t)

        x_start = np.concatenate([a_best, [best_score * 0.95]])
        bounds = [(0, 2 * np.pi)] * n + [(0.0, 1.0)]

        try:
            result2 = minimize(_epigraph_slsqp, x_start, method='SLSQP',
                             bounds=bounds,
                             options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            if result2.fun < 0:
                a = np.sort(result2.x[:-1] % (2 * np.pi))
                poly = np.stack([np.cos(a), np.sin(a)], axis=1)
                if _is_valid_convex(poly):
                    exact_s = _polygon_score(poly)
                    if exact_s > best_score:
                        best_score = exact_s
                        best_polygon = poly
        except Exception:
            pass

    # Phase 3: Vertex-level targeted refinement
    # Focus on the worst vertex and try to improve it
    if _elapsed() < deadline - 60:
        scores = _all_vertex_scores(best_polygon)
        worst_idx = int(np.argmin(scores))
        worst_score = scores[worst_idx]

        # Try perturbing neighbors of worst vertex to improve its score
        a_best = np.arctan2(best_polygon[:, 1], best_polygon[:, 0])
        a_best = np.sort(a_best % (2 * np.pi))

        for attempt in range(5):
            if _elapsed() > deadline - 30:
                break
            # Perturb a random subset of angles near the worst vertex
            x_start = a_best.copy()
            perturbation_mask = rng.random(n) < 0.3
            x_start[perturbation_mask] += rng.normal(0, 0.01, n)

            try:
                result3 = minimize(_neg_softmin_score, x_start, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                if result3.fun < 0:
                    a = np.sort(result3.x % (2 * np.pi))
                    poly = np.stack([np.cos(a), np.sin(a)], axis=1)
                    if _is_valid_convex(poly):
                        exact_s = _polygon_score(poly)
                        if exact_s > best_score:
                            best_score = exact_s
                            best_polygon = poly
                            # Re-identify worst vertex
                            scores = _all_vertex_scores(best_polygon)
                            worst_idx = int(np.argmin(scores))
                            worst_score = scores[worst_idx]
            except Exception:
                pass

    # Phase 4: Final high-precision polish
    if _elapsed() < deadline - 20:
        a_best = np.arctan2(best_polygon[:, 1], best_polygon[:, 0])
        a_best = np.sort(a_best % (2 * np.pi))

        try:
            result4 = minimize(_neg_softmin_score, a_best, method='Nelder-Mead',
                             options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True})
            if result4.fun < 0:
                a = np.sort(result4.x % (2 * np.pi))
                poly = np.stack([np.cos(a), np.sin(a)], axis=1)
                if _is_valid_convex(poly):
                    exact_s = _polygon_score(poly)
                    if exact_s > best_score:
                        best_score = exact_s
                        best_polygon = poly
        except Exception:
            pass

    # Final normalization and validity check
    best_polygon = _normalize(best_polygon)
    if not _is_valid_convex(best_polygon):
        # Fall back to regular polygon
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        best_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        best_polygon = _normalize(best_polygon)

    return best_polygon


# EVOLVE-BLOCK-END
