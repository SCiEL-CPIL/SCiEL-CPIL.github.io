import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

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
        four_d = d[k-1:k+3]
    if np.any(four_d <= 0):
        return 0.0
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return float(score)

def _soft_min_vertex_score(polygon, temperature=0.02):
    """Smooth surrogate for the minimum vertex score using log-sum-exp."""
    n = len(polygon)
    if n != 20:
        return 0.0
    if not np.all(np.isfinite(polygon)):
        return 0.0
    centroid = np.mean(polygon, axis=0)
    p = polygon - centroid
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist

    # Check convexity
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i+1) % n] - p[i]
        e2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    if np.any(np.abs(cross_products) < 1e-10):
        return 0.0
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return 0.0

    # Vectorized pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    scores = np.zeros(n)
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        scores[i] = _vertex_score(d_sorted)

    # Soft min via log-sum-exp (shift for numerical stability)
    min_s = np.min(scores)
    if min_s <= 0:
        return 0.0
    exp_scores = np.exp(-(scores - min_s) / temperature)
    soft_min = min_s - temperature * np.log(np.sum(exp_scores))
    return float(soft_min)

def _is_valid_convex(p):
    n = len(p)
    if not np.all(np.isfinite(p)):
        return False
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-10:
                return False
    cross_products = np.zeros(n)
    for i in range(n):
        e1 = p[(i+1) % n] - p[i]
        e2 = p[(i+2) % n] - p[(i+1) % n]
        cross_products[i] = e1[0] * e2[1] - e1[1] * e2[0]
    if np.any(np.abs(cross_products) < 1e-14):
        return False
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
    total_turn = 0.0
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
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
    dists_to_centroid = np.sqrt(np.sum(p**2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    p = p / mean_dist

    if not _is_valid_convex(p):
        return 0.0

    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

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

def _angular_to_poly(angles):
    """Convert angles to polygon vertices on unit circle."""
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)

def _fourier_to_poly(params, base_angles):
    """Convert Fourier coefficients to polygon vertices."""
    # params: [a1, b1, a2, b2, ..., aM, bM] for harmonics 1..M
    angles = base_angles.copy()
    n_harmonics = len(params) // 2
    for m in range(1, n_harmonics + 1):
        a = params[2*(m-1)]
        b = params[2*(m-1)+1]
        angles += a * np.cos(m * base_angles) + b * np.sin(m * base_angles)
    return _angular_to_poly(angles)

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)

    # --- Strategy 1: Differential Evolution with Fourier basis ---
    def _objective_fourier(params):
        poly = _fourier_to_poly(params, base_angles)
        s_actual = _compute_min_vertex_score(poly)
        if s_actual > 0:
            return -s_actual
        else:
            s_soft = _soft_min_vertex_score(poly, temperature=0.02)
            return 1.0 - s_soft

    # Use 8 harmonics for richer representation
    n_harmonics = 8
    bounds_fourier = [(-0.15, 0.15)] * (2 * n_harmonics)
    
    try:
        result_de = differential_evolution(
            _objective_fourier,
            bounds_fourier,
            maxiter=200,
            popsize=30,
            tol=1e-12,
            seed=42,
            polish=True
        )
        score = -result_de.fun if result_de.fun < 0 else 0.0
        if score > best_score + 1e-9:
            best_score = score
            best_poly = _fourier_to_poly(result_de.x, base_angles)
    except Exception:
        pass

    if time.time() > deadline - 120:
        return best_poly

    # --- Strategy 2: Direct angular optimization with Nelder-Mead ---
    def _angular_poly_direct(params):
        angles = base_angles + params
        return _angular_to_poly(angles)

    def _objective_angular_direct(params):
        poly = _angular_poly_direct(params)
        s_actual = _compute_min_vertex_score(poly)
        if s_actual > 0:
            return -s_actual
        else:
            s_soft = _soft_min_vertex_score(poly, temperature=0.01)
            return 1.0 - s_soft

    # Start from current best
    current_angles = np.arctan2(best_poly[:, 1], best_poly[:, 0])
    current_angles = np.unwrap(current_angles)
    angle_params = current_angles - base_angles
    angle_params = (angle_params + np.pi) % (2 * np.pi) - np.pi

    try:
        result_nm = minimize(
            _objective_angular_direct,
            angle_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
        )
        score = -result_nm.fun if result_nm.fun < 0 else 0.0
        if score > best_score + 1e-9:
            best_score = score
            best_poly = _angular_poly_direct(result_nm.x)
    except Exception:
        pass

    if time.time() > deadline - 80:
        return best_poly

    # --- Strategy 3: Multi-start Nelder-Mead on angular params ---
    def _multi_start_angular():
        nonlocal best_poly, best_score
        rng = np.random.RandomState(123)
        for restart in range(12):
            if time.time() > deadline - 60:
                break
            # Perturb current best angles
            current_angles = np.arctan2(best_poly[:, 1], best_poly[:, 0])
            current_angles = np.unwrap(current_angles)
            angle_params = current_angles - base_angles
            angle_params = (angle_params + np.pi) % (2 * np.pi) - np.pi
            # Add random perturbation
            perturbation = rng.normal(0, 0.015 * (1 + restart*0.2), size=20)
            x0 = angle_params + perturbation
            
            try:
                result = minimize(
                    _objective_angular_direct,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
                )
                score = -result.fun if result.fun < 0 else 0.0
                if score > best_score + 1e-9:
                    best_score = score
                    best_poly = _angular_poly_direct(result.x)
            except Exception:
                pass

    _multi_start_angular()

    if time.time() > deadline - 40:
        return best_poly

    # --- Strategy 4: High-precision angular polish with exact objective ---
    def _objective_angular_exact(params):
        poly = _angular_poly_direct(params)
        return -_compute_min_vertex_score(poly)

    # Get current best angles
    current_angles = np.arctan2(best_poly[:, 1], best_poly[:, 0])
    current_angles = np.unwrap(current_angles)
    angle_params = current_angles - base_angles
    angle_params = (angle_params + np.pi) % (2 * np.pi) - np.pi

    try:
        result = minimize(
            _objective_angular_exact,
            angle_params,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True}
        )
        score = -result.fun if result_nm.fun < 0 else 0.0
        if score > best_score + 1e-9:
            best_score = score
            best_poly = _angular_poly_direct(result.x)
    except Exception:
        pass

    if time.time() > deadline - 20:
        return best_poly

    # --- Strategy 5: Final L-BFGS-B polish in angular space ---
    def _objective_angular_lbfgs(params):
        poly = _angular_poly_direct(params)
        s = _compute_min_vertex_score(poly)
        if s <= 0:
            return 1.0 - _soft_min_vertex_score(poly, temperature=0.005)
        return -s

    current_angles = np.arctan2(best_poly[:, 1], best_poly[:, 0])
    current_angles = np.unwrap(current_angles)
    angle_params = current_angles - base_angles
    angle_params = (angle_params + np.pi) % (2 * np.pi) - np.pi

    try:
        result = minimize(
            _objective_angular_lbfgs,
            angle_params,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-20, 'gtol': 1e-18, 'maxls': 300}
        )
        score = -result.fun if result.fun < 0 else 0.0
        if score > best_score + 1e-9:
            best_score = score
            best_poly = _angular_poly_direct(result.x)
    except Exception:
        pass

    # Final validation
    final_score = _compute_min_vertex_score(best_poly)
    if final_score > 0:
        return best_poly
    else:
        # Fallback to regular polygon
        return _regular_polygon()
