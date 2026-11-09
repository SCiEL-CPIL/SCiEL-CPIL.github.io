# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _vertex_score(distances_sorted):
    d = distances_sorted
    n = len(d)
    if n != 19:
        return 0.0
    if any(di < 0.05 for di in d):
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
    ratios = np.maximum(four_d / D, D / four_d)
    raw = np.mean(ratios)
    score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
    return float(score)

def _is_valid_convex(p):
    """Fast convexity check for normalized polygon."""
    n = len(p)
    if not np.all(np.isfinite(p)):
        return False
    # Check no repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.allclose(p[i], p[j], atol=1e-10):
                return False
    # Cross products all same sign
    cross_products = np.array([
        (p[(i+1)%n] - p[i])[0] * (p[(i+2)%n] - p[(i+1)%n])[1] -
        (p[(i+1)%n] - p[i])[1] * (p[(i+2)%n] - p[(i+1)%n])[0]
        for i in range(n)
    ])
    if np.any(np.abs(cross_products) < 1e-12):
        return False
    signs = np.sign(cross_products)
    if not np.all(signs == signs[0]):
        return False
    # Total turning angle
    total_turn = 0.0
    for i in range(n):
        v1 = p[(i+1) % n] - p[i]
        v2 = p[(i+2) % n] - p[(i+1) % n]
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        turn = angle2 - angle1
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
    
    # Vectorized distance computation
    # dists[i,j] = distance from p[i] to p[j]
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    min_score = 1.0
    for i in range(n):
        d = np.delete(dists[i], i)
        d_sorted = np.sort(d)
        score = _vertex_score(d_sorted)
        min_score = min(min_score, score)
    return min_score

def _negative_score_for_optimization(flat_params):
    """Objective function for scipy minimize: return negative min vertex score."""
    poly = flat_params.reshape(20, 2)
    score = _compute_min_vertex_score(poly)
    return -score

def _regular_polygon():
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.stack([x, y], axis=1)

def _optimize_with_scipy(initial_poly, max_time=300.0):
    """
    Use scipy.optimize.minimize with Nelder-Mead to optimize the polygon.
    """
    start_time = time.time()
    best_poly = initial_poly.copy()
    best_score = _compute_min_vertex_score(best_poly)
    
    x0 = initial_poly.flatten()
    
    # Use Nelder-Mead which doesn't require gradients
    result = minimize(
        _negative_score_for_optimization,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    
    if result.fun < -best_score:
        best_poly = result.x.reshape(20, 2)
        best_score = -result.fun
    
    # Second pass with tighter tolerances
    if time.time() - start_time < max_time * 0.7:
        x0 = best_poly.flatten()
        result2 = minimize(
            _negative_score_for_optimization,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if result2.fun < -best_score:
            best_poly = result2.x.reshape(20, 2)
            best_score = -result2.fun
    
    return best_poly, best_score

def _try_symmetric_patterns():
    """
    Try various symmetric patterns to find good starting points.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    # Radial modulation: r_i = 1 + a * cos(m * theta_i)
    for a in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]:
        for m in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            radii = 1.0 + a * np.cos(m * base_angles)
            x = radii * np.cos(base_angles)
            y = radii * np.sin(base_angles)
            poly = np.stack([x, y], axis=1)
            score = _compute_min_vertex_score(poly)
            if score > best_score:
                best_score = score
                best_poly = poly.copy()
    
    # Sinusoidal radial modulation
    for a in [0.01, 0.02, 0.03, 0.05]:
        for m in [1, 2, 3, 4, 5]:
            radii = 1.0 + a * np.sin(m * base_angles)
            x = radii * np.cos(base_angles)
            y = radii * np.sin(base_angles)
            poly = np.stack([x, y], axis=1)
            score = _compute_min_vertex_score(poly)
            if score > best_score:
                best_score = score
                best_poly = poly.copy()
    
    # Angular modulation
    for a in [0.005, 0.01, 0.02, 0.03, 0.05]:
        for m in [1, 2, 3, 4, 5]:
            angles = base_angles + a * np.sin(m * base_angles)
            order = np.argsort(angles)
            angles_sorted = angles[order]
            x = np.cos(angles_sorted)
            y = np.sin(angles_sorted)
            poly = np.stack([x, y], axis=1)
            score = _compute_min_vertex_score(poly)
            if score > best_score:
                best_score = score
                best_poly = poly.copy()
    
    # Combined radial and angular
    for ra in [0.01, 0.02, 0.03]:
        for rm in [1, 2, 3]:
            for aa in [0.01, 0.02, 0.03]:
                for am in [1, 2, 3]:
                    radii = 1.0 + ra * np.cos(rm * base_angles)
                    angles = base_angles + aa * np.sin(am * base_angles)
                    order = np.argsort(angles)
                    angles_sorted = angles[order]
                    radii_sorted = radii[order]
                    x = radii_sorted * np.cos(angles_sorted)
                    y = radii_sorted * np.sin(angles_sorted)
                    poly = np.stack([x, y], axis=1)
                    score = _compute_min_vertex_score(poly)
                    if score > best_score:
                        best_score = score
                        best_poly = poly.copy()
    
    return best_poly, best_score

def _coordinate_descent_refine(poly, max_time=60.0):
    """Quick coordinate descent refinement after scipy optimization."""
    start_time = time.time()
    current_poly = poly.copy()
    current_score = _compute_min_vertex_score(current_poly)
    best_poly = current_poly.copy()
    best_score = current_score
    
    step_size = 0.02
    no_improve = 0
    
    while time.time() - start_time < max_time and no_improve < 20:
        improved = False
        for i in range(20):
            if time.time() - start_time > max_time:
                break
            old_v = current_poly[i].copy()
            for dx in [-step_size, step_size]:
                for dy in [-step_size, step_size]:
                    new_poly = current_poly.copy()
                    new_poly[i] = old_v + np.array([dx, dy])
                    score = _compute_min_vertex_score(new_poly)
                    if score > current_score + 1e-12:
                        current_poly = new_poly
                        current_score = score
                        improved = True
                        if score > best_score:
                            best_score = score
                            best_poly = current_poly.copy()
                        break
                if improved:
                    break
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            step_size *= 0.6
    
    return best_poly, best_score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 595.0  # Safety margin before 600s
    
    # Start with regular polygon
    best_poly = _regular_polygon()
    best_score = _compute_min_vertex_score(best_poly)
    
    if time.time() > deadline:
        return best_poly
    
    # Try symmetric patterns to find good starting point
    sym_poly, sym_score = _try_symmetric_patterns()
    if sym_score > best_score:
        best_score = sym_score
        best_poly = sym_poly
    
    if time.time() > deadline:
        return best_poly
    
    # Scipy optimization from best starting point
    scipy_time = min(250.0, deadline - time.time() - 30.0)
    if scipy_time > 30:
        sc_poly, sc_score = _optimize_with_scipy(best_poly, max_time=scipy_time)
        if sc_score > best_score:
            best_score = sc_score
            best_poly = sc_poly
    
    if time.time() > deadline:
        return best_poly
    
    # Coordinate descent refinement
    cd_time = min(60.0, deadline - time.time() - 5.0)
    if cd_time > 5:
        cd_poly, cd_score = _coordinate_descent_refine(best_poly, max_time=cd_time)
        if cd_score > best_score:
            best_score = cd_score
            best_poly = cd_poly
    
    return best_poly

# EVOLVE-BLOCK-END
