import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _vertex_score(distances_sorted):
    d = distances_sorted
    n = len(d)
    if n != 19:
        return 0.0
    if np.any(d < 0.05):
        return 0.0
    diffs = d[1:] - d[:-1]
    min_diff = np.min(diffs)
    k = int(np.argmax(diffs == min_diff))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        four_d = d[0:4]
    elif k == 17:
        four_d = d[15:19]
    else:
        four_d = d[k - 1:k + 3]
    if np.any(four_d <= 0):
        return 0.0
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return float(score)

def _is_valid_convex(p):
    n = len(p)
    if not np.all(np.isfinite(p)):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return False
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i + 1) % n] - p[i]
        e2 = p[(i + 2) % n] - p[(i + 1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    if np.any(np.abs(cross_products) < 1e-12):
        return False
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
    total_turn = 0.0
    for i in range(n):
        v1 = p[(i + 1) % n] - p[i]
        v2 = p[(i + 2) % n] - p[(i + 1) % n]
        ang1 = np.arctan2(v1[1], v1[0])
        ang2 = np.arctan2(v2[1], v2[0])
        turn = ang2 - ang1
        while turn > np.pi:
            turn -= 2 * np.pi
        while turn < -np.pi:
            turn += 2 * np.pi
        total_turn += turn
    if not np.isclose(abs(total_turn), 2 * np.pi, atol=1e-6):
        return False
    return True

def _compute_min_vertex_score(polygon):
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist
    if not _is_valid_convex(p):
        return 0.0
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    min_score = 1.0
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        min_score = min(min_score, score)
    return min_score

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def _superellipse_polygon(p=4.0):
    n = 20
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.sign(np.cos(t)) * (np.abs(np.cos(t)) ** (2.0 / p))
    y = np.sign(np.sin(t)) * (np.abs(np.sin(t)) ** (2.0 / p))
    return np.stack([x, y], axis=1)

def _soft_min_vertex_score(polygon, temperature=0.05):
    """Smooth surrogate for minimum vertex score using log-sum-exp."""
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist

    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i + 1) % n] - p[i]
        e2 = p[(i + 2) % n] - p[(i + 1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    if np.any(np.abs(cross_products) < 1e-12):
        return 0.0
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return 0.0

    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))

    scores = np.zeros(n)
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        scores[i] = _vertex_score(d_sorted)

    max_s = np.max(scores)
    if max_s <= 0:
        return 0.0
    exp_scores = np.exp(-scores / temperature)
    soft_min = -temperature * np.log(np.sum(exp_scores))
    return float(soft_min)

def _convexity_penalty(polygon, weight=1.0):
    n = len(polygon)
    if n != 20:
        return 1.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 1.0
    p = p / mean_dist

    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i + 1) % n] - p[i]
        e2 = p[(i + 2) % n] - p[(i + 1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    neg_cross = np.maximum(0, -cross_products)
    penalty = np.sum(neg_cross ** 2)
    return weight * penalty

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 610.0  # safety margin before 630s kill

    n = 20
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)

    # Initialize with superellipse candidates
    for p_val in [3.0, 4.0, 6.0, 8.0, 12.0]:
        if time.time() > deadline - 10:
            break
        poly = _superellipse_polygon(p_val)
        score = _compute_min_vertex_score(poly)
        if score > best_score + 1e-9:
            best_score = score
            best_poly = poly.copy()

    if time.time() > deadline - 10:
        return best_poly

    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    # --- Phase 1: Differential Evolution over angular perturbations ---
    # This is the structural idea from the inspiration: DE explores a bounded
    # parameter space (angular deviations) that naturally preserves cyclic order
    # and convexity, letting it focus on the shape that improves equidistance.
    def _angular_poly(params):
        angles = base_angles + params
        angles = np.sort(angles)
        angles = angles - np.min(angles)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)

    def _objective_angular(params):
        poly = _angular_poly(params)
        s_actual = _compute_min_vertex_score(poly)
        if s_actual > 0:
            return -s_actual
        s_soft = _soft_min_vertex_score(poly, temperature=0.05)
        return 1.0 - s_soft if s_soft > 0 else 1.0

    bounds = [(-0.12, 0.12)] * n
    try:
        result_de = differential_evolution(
            _objective_angular,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True,
            init='sobol'
        )
        poly = _angular_poly(result_de.x)
        score = _compute_min_vertex_score(poly)
        if score > best_score + 1e-9:
            best_score = score
            best_poly = poly.copy()
    except Exception:
        pass

    if time.time() > deadline - 150:
        return best_poly

    # --- Phase 2: Radial Perturbation Search ---
    # Structural change: fix the angular structure (sorted angles from best),
    # and optimize the RADIAL distance of each vertex from the centroid.
    # This is a different representation family from pure angular or pure
    # Cartesian search: it decouples "where around the circle" from "how far out",
    # enabling superellipse-like and non-circular shapes while preserving order.
    best_angles_current = np.arctan2(best_poly[:, 1], best_poly[:, 0])
    order = np.argsort(best_angles_current)
    best_angles_sorted = best_angles_current[order]
    best_angles_sorted = best_angles_sorted - best_angles_sorted[0]

    def _radial_poly(rad_params):
        # rad_params: radial scale per vertex (normalized to mean=1)
        r = 1.0 + rad_params
        r = np.clip(r, 0.3, 3.0)
        x = r * np.cos(best_angles_sorted)
        y = r * np.sin(best_angles_sorted)
        return np.stack([x, y], axis=1)

    def _objective_radial(rad_params):
        poly = _radial_poly(rad_params)
        s_actual = _compute_min_vertex_score(poly)
        if s_actual > 0:
            return -s_actual
        s_soft = _soft_min_vertex_score(poly, temperature=0.05)
        pen = _convexity_penalty(poly, weight=0.5)
        return 1.0 - s_soft + pen if s_soft > 0 else 1.0 + pen

    rad_bounds = [(-0.25, 0.25)] * n
    try:
        result_de2 = differential_evolution(
            _objective_radial,
            rad_bounds,
            maxiter=80,
            popsize=15,
            tol=1e-8,
            seed=7,
            polish=True,
            init='sobol'
        )
        poly = _radial_poly(result_de2.x)
        score = _compute_min_vertex_score(poly)
        if score > best_score + 1e-9:
            best_score = score
            best_poly = poly.copy()
    except Exception:
        pass

    if time.time() > deadline - 100:
        return best_poly

    # --- Phase 3: General Coordinate Search (Non-Cyclic) with soft surrogate ---
    def _objective_coord(params, temp=0.05, pen_weight=1.0):
        p = params.reshape(n, 2)
        if not np.all(np.isfinite(p)):
            return 100.0
        centroid = np.mean(p, axis=0)
        p_centered = p - centroid
        dists = np.sqrt(np.sum(p_centered ** 2, axis=1))
        mean_dist = np.mean(dists)
        if mean_dist <= 0.01:
            return 100.0
        pen = _convexity_penalty(p, weight=pen_weight)
        if pen > 1.0:
            return 10.0 + pen
        s = _soft_min_vertex_score(p, temperature=temp)
        return -s + pen

    x0 = best_poly.flatten()
    try:
        result = minimize(
            _objective_coord,
            x0,
            args=(0.05, 1.0),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        p_opt = result.x.reshape(n, 2)
        score = _compute_min_vertex_score(p_opt)
        if score > best_score + 1e-9:
            best_score = score
            best_poly = p_opt.copy()
    except Exception:
        pass

    if time.time() > deadline - 60:
        return best_poly

    # --- Phase 4: Nelder-Mead polish in Cartesian space (hard score) ---
    def _objective_hard_coord(params):
        p = params.reshape(n, 2)
        if not np.all(np.isfinite(p)):
            return 100.0
        centroid = np.mean(p, axis=0)
        p_centered = p - centroid
        dists = np.sqrt(np.sum(p_centered ** 2, axis=1))
        mean_dist = np.mean(dists)
        if mean_dist <= 0.01:
            return 100.0
        cross_products = np.zeros(n)
        for i in range(n):
            e1 = p_centered[(i + 1) % n] - p_centered[i]
            e2 = p_centered[(i + 2) % n] - p_centered[(i + 1) % n]
            cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
        if not np.all(cross_products > 1e-8):
            return 10.0 + np.sum(np.maximum(0, -cross_products) ** 2)
        s = _compute_min_vertex_score(p)
        return -s if s > 0 else 10.0

    try:
        result = minimize(
            _objective_hard_coord,
            best_poly.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        p_opt = result.x.reshape(n, 2)
        score = _compute_min_vertex_score(p_opt)
        if score > best_score + 1e-9:
            best_score = score
            best_poly = p_opt.copy()
    except Exception:
        pass

    if time.time() > deadline - 30:
        return best_poly

    # --- Phase 5: Local coordinate descent with adaptive step (refinement) ---
    current_poly = best_poly.copy()
    current_score = best_score
    step_size = 0.01
    no_improve = 0

    while time.time() < deadline - 10 and no_improve < 40:
        improved = False
        for i in range(n):
            if time.time() > deadline - 10:
                break
            old_v = current_poly[i].copy()
            best_local = current_score
            best_new = None
            for dx in [-step_size, -step_size * 0.5, 0.0, step_size * 0.5, step_size]:
                for dy in [-step_size, -step_size * 0.5, 0.0, step_size * 0.5, step_size]:
                    if dx == 0.0 and dy == 0.0:
                        continue
                    new_poly = current_poly.copy()
                    new_poly[i] = old_v + np.array([dx, dy])
                    s = _compute_min_vertex_score(new_poly)
                    if s > best_local + 1e-9:
                        best_local = s
                        best_new = new_poly.copy()
            if best_new is not None:
                current_poly = best_new
                current_score = best_local
                improved = True
        if not improved:
            no_improve += 1
            step_size *= 0.7
        else:
            no_improve = 0
            step_size = max(0.001, step_size * 0.85)

    if current_score > best_score + 1e-9:
        best_score = current_score
        best_poly = current_poly.copy()

    return best_poly
