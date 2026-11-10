import numpy as np
import time
from scipy.optimize import minimize

_N = 20
_BASE_ANGLES = np.linspace(0, 2 * np.pi, _N, endpoint=False)


def _build_points(radii, angle_deltas=None):
    angles = _BASE_ANGLES
    if angle_deltas is not None:
        angles = angles + angle_deltas
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))


def _normalize(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return None
    return points / mean_dist


def _cross_products(points):
    p_next = np.roll(points, -1, axis=0)
    p_next2 = np.roll(points, -2, axis=0)
    e1 = p_next - points
    e2 = p_next2 - p_next
    return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]


def _compute_scores(points):
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    dists_self_inf = dists.copy()
    np.fill_diagonal(dists_self_inf, np.inf)
    sorted_d = np.sort(dists_self_inf, axis=1)
    d = sorted_d[:, :n - 1]

    scores = np.zeros(n)
    has_close = np.any(d < 0.05, axis=1)
    gaps = d[:, 1:] - d[:, :-1]
    k_idx = np.argmin(gaps, axis=1)
    D = (d[np.arange(n), k_idx] + d[np.arange(n), k_idx + 1]) / 2.0

    valid = (~has_close) & (D >= 0.1)

    n_minus_3 = n - 3
    selected = np.zeros((n, 4))

    mask_k0 = valid & (k_idx == 0)
    mask_k17 = valid & (k_idx == n_minus_3)
    mask_mid = valid & (k_idx > 0) & (k_idx < n_minus_3)

    for i in np.where(mask_k0)[0]:
        selected[i] = d[i, 0:4]
    for i in np.where(mask_k17)[0]:
        selected[i] = d[i, 15:19]
    for i in np.where(mask_mid)[0]:
        k = k_idx[i]
        selected[i] = d[i, k - 1:k + 3]

    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.maximum(selected / D[:, np.newaxis], D[:, np.newaxis] / selected)
    raw = np.mean(ratios, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        score_vals = np.minimum(1.0, 1.0 / raw)
    scores = np.where(valid, score_vals, 0.0)
    return scores


def _strictly_convex(points):
    cross = _cross_products(points)
    if np.any(cross == 0):
        return False
    if np.any(cross > 0) and np.any(cross < 0):
        return False
    if np.min(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)) < 1e-9:
        return False
    return True


def _support_func_to_polygon(coeffs, n_harmonics=4):
    """
    Convert Fourier coefficients of support function to polygon vertices.
    h(theta) = a_0 + sum_{k=1}^{n_h} (a_k cos(k*theta) + b_k sin(k*theta))
    coeffs = [a_0, a_1, b_1, a_2, b_2, ..., a_{n_h}, b_{n_h}]
    The polygon is the intersection of half-planes.
    """
    theta = np.linspace(0, 2 * np.pi, _N, endpoint=False)
    
    a_0 = coeffs[0]
    h = np.full(_N, a_0)
    for k in range(1, n_harmonics + 1):
        a_k = coeffs[2 * k - 1]
        b_k = coeffs[2 * k]
        h += a_k * np.cos(k * theta) + b_k * np.sin(k * theta)
    
    # Ensure h is positive (polygon contains origin)
    if np.min(h) < 0.05:
        h += 0.05 - np.min(h)
    
    # Compute vertices as intersection of consecutive half-planes
    vertices = np.zeros((_N, 2))
    for i in range(_N):
        i_next = (i + 1) % _N
        
        c1, s1 = np.cos(theta[i]), np.sin(theta[i])
        c2, s2 = np.cos(theta[i_next]), np.sin(theta[i_next])
        
        det = c1 * s2 - s1 * c2
        if abs(det) < 1e-10:
            vertices[i] = [(h[i] + h[i_next]) / 2, 0]
            continue
        
        x = (h[i] * s2 - h[i_next] * s1) / det
        y = (c1 * h[i_next] - c2 * h[i]) / det
        vertices[i] = [x, y]
    
    return vertices


def _min_vertex_score_from_points(points, check_convexity=True):
    if points is None:
        return 0.0
    if check_convexity and not _strictly_convex(points):
        return 0.0
    return float(np.min(_compute_scores(points)))


def _score_support(params, n_harmonics=4):
    """Objective for support function optimization. Returns negative min score."""
    polygon = _support_func_to_polygon(params, n_harmonics)
    polygon_norm = _normalize(polygon)
    if polygon_norm is None:
        return 1000.0
    
    # Support function guarantees convexity, but verify
    cross = _cross_products(polygon_norm)
    if np.any(cross <= 0):
        # Not strictly convex - add penalty
        penalty = np.sum(np.maximum(0, -cross) ** 2)
        return 100.0 + penalty
    
    scores = _compute_scores(polygon_norm)
    min_score = np.min(scores)
    return -min_score


def _smooth_score_support(params, n_harmonics=4, beta=30.0):
    """Smooth surrogate using soft-min for support function optimization."""
    polygon = _support_func_to_polygon(params, n_harmonics)
    polygon_norm = _normalize(polygon)
    if polygon_norm is None:
        return 1000.0
    
    cross = _cross_products(polygon_norm)
    pen = float(np.sum(np.maximum(0.0, -cross) ** 2))
    if pen > 1e-3:
        return 100.0 + pen
    
    scores = _compute_scores(polygon_norm)
    s_clamped = np.clip(scores, 1e-8, 1.0)
    neg_beta_s = -beta * s_clamped
    max_neg = np.max(neg_beta_s)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_s - max_neg)))
    soft_min = -log_sum / beta
    return -soft_min


def _score_radial(params, check_convexity=True):
    """Objective for radial parameter optimization."""
    radii = np.maximum(params, 0.05)
    points = _normalize(_build_points(radii))
    if points is None:
        return 1000.0
    if check_convexity:
        cross = _cross_products(points)
        pen = float(np.sum(np.maximum(0.0, -cross) ** 2))
        if pen > 1e-3:
            return 100.0 + pen
    scores = _compute_scores(points)
    return -np.min(scores)


def _smooth_score_radial(params, beta=30.0):
    """Smooth surrogate for radial parameters."""
    radii = np.maximum(params, 0.05)
    points = _normalize(_build_points(radii))
    if points is None:
        return 1000.0
    cross = _cross_products(points)
    pen = float(np.sum(np.maximum(0.0, -cross) ** 2))
    if pen > 1e-3:
        return 100.0 + pen
    scores = _compute_scores(points)
    s_clamped = np.clip(scores, 1e-8, 1.0)
    neg_beta_s = -beta * s_clamped
    max_neg = np.max(neg_beta_s)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_s - max_neg)))
    soft_min = -log_sum / beta
    return -soft_min


def _score_joint(params, check_convexity=True):
    """Objective for joint (radii + angle_deltas) optimization."""
    n = _N
    radii = np.maximum(params[:n], 0.05)
    angle_deltas = params[n:]
    points = _normalize(_build_points(radii, angle_deltas))
    if points is None:
        return 1000.0
    if check_convexity:
        cross = _cross_products(points)
        pen = float(np.sum(np.maximum(0.0, -cross) ** 2))
        if pen > 1e-3:
            return 100.0 + pen
    scores = _compute_scores(points)
    return -np.min(scores)


def _smooth_score_joint(params, beta=30.0):
    """Smooth surrogate for joint parameters."""
    n = _N
    radii = np.maximum(params[:n], 0.05)
    angle_deltas = params[n:]
    points = _normalize(_build_points(radii, angle_deltas))
    if points is None:
        return 1000.0
    cross = _cross_products(points)
    pen = float(np.sum(np.maximum(0.0, -cross) ** 2))
    if pen > 1e-3:
        return 100.0 + pen
    scores = _compute_scores(points)
    s_clamped = np.clip(scores, 1e-8, 1.0)
    neg_beta_s = -beta * s_clamped
    max_neg = np.max(neg_beta_s)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_s - max_neg)))
    soft_min = -log_sum / beta
    return -soft_min


def _extract_best_from_support(params, n_harmonics=4):
    """Extract normalized polygon from support function params."""
    polygon = _support_func_to_polygon(params, n_harmonics)
    return _normalize(polygon)


def _extract_best_from_radial(params):
    """Extract normalized polygon from radial params."""
    radii = np.maximum(params, 0.05)
    polygon = _build_points(radii)
    return _normalize(polygon)


def _extract_best_from_joint(params):
    """Extract normalized polygon from joint params."""
    n = _N
    radii = np.maximum(params[:n], 0.05)
    angle_deltas = params[n:]
    polygon = _build_points(radii, angle_deltas)
    return _normalize(polygon)


def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0
    
    best_score = -1.0
    best_points = None
    
    def try_update(points):
        nonlocal best_score, best_points
        if points is None:
            return
        if not _strictly_convex(points):
            return
        sc = float(np.min(_compute_scores(points)))
        if sc > best_score:
            best_score = sc
            best_points = points.copy()
    
    # Probe evaluation cost
    probe_start = time.time()
    _score_support(np.ones(9))
    eval_time = max(time.time() - probe_start, 1e-4)
    
    # --- Phase 1: Support function multi-start with smooth objective ---
    # Support function guarantees convexity, so we can focus on score
    n_harm = 4
    n_params = 2 * n_harm + 1  # a_0, (a_k, b_k) for k=1..n_harm
    
    rng = np.random.default_rng(42)
    
    # Initializations for support function
    support_inits = []
    # Uniform (circle)
    support_inits.append(np.concatenate([[1.0], np.zeros(2 * n_harm)]))
    # Small perturbations
    for _ in range(8):
        init = np.concatenate([[1.0], 0.05 * rng.standard_normal(2 * n_harm)])
        support_inits.append(init)
    # Fourier modes
    for mode in [1, 2, 3, 4]:
        for amp in [0.05, 0.1, 0.15]:
            init = np.zeros(n_params)
            init[0] = 1.0
            init[2 * mode - 1] = amp  # a_mode
            support_inits.append(init.copy())
            init2 = np.zeros(n_params)
            init2[0] = 1.0
            init2[2 * mode] = amp  # b_mode
            support_inits.append(init2.copy())
    
    # Optimize support function inits
    for i, x0 in enumerate(support_inits):
        if time.time() > deadline - 150:
            break
        remaining = deadline - time.time()
        # Smooth phase
        max_iter = min(2000, max(200, int(remaining * 0.15 / max(eval_time * n_params, 1e-3))))
        try:
            res = minimize(lambda p: _smooth_score_support(p, n_harm, beta=25.0),
                          x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
            points = _extract_best_from_support(res.x, n_harm)
            try_update(points)
        except Exception:
            pass
        
        if time.time() > deadline - 130:
            break
        
        # Sharp phase
        max_iter2 = min(2000, max(150, int(remaining * 0.10 / max(eval_time * n_params, 1e-3))))
        try:
            res2 = minimize(lambda p: _score_support(p, n_harm),
                           res.x, method='Nelder-Mead',
                           options={'maxiter': max_iter2, 'xatol': 1e-12, 'fatol': 1e-12})
            points2 = _extract_best_from_support(res2.x, n_harm)
            try_update(points2)
        except Exception:
            pass
    
    # --- Phase 2: Radial parameter optimization ---
    a = _BASE_ANGLES
    rng_a = np.random.default_rng(12345)
    rng_b = np.random.default_rng(67890)
    
    radial_inits = [
        np.ones(_N),
        np.maximum(1.0 + 0.15 * np.cos(2 * a), 0.05),
        np.maximum(1.0 + 0.15 * np.cos(4 * a), 0.05),
        np.maximum(1.0 + 0.15 * np.cos(5 * a), 0.05),
        np.maximum(1.0 + 0.10 * np.cos(3 * a), 0.05),
        np.maximum(1.0 + 0.10 * np.cos(6 * a), 0.05),
        np.maximum(1.0 + 0.10 * np.cos(7 * a), 0.05),
        np.maximum(1.0 + 0.25 * np.cos(4 * a), 0.05),
        np.maximum(1.0 + 0.25 * np.cos(5 * a), 0.05),
        np.maximum(1.0 + 0.08 * rng_a.standard_normal(_N), 0.05),
        np.maximum(1.0 + 0.12 * rng_b.standard_normal(_N), 0.05),
        np.maximum(1.0 + 0.12 * np.cos(2 * a) + 0.08 * np.cos(4 * a), 0.05),
        np.maximum(1.0 + 0.10 * np.cos(3 * a) + 0.06 * np.cos(6 * a), 0.05),
    ]
    alt = np.ones(_N)
    alt[::2] = 1.1
    alt[1::2] = 0.9
    radial_inits.append(alt)
    
    for x0 in radial_inits:
        if time.time() > deadline - 100:
            break
        remaining = deadline - time.time()
        max_iter = min(2000, max(200, int(remaining * 0.12 / max(eval_time * _N, 1e-3))))
        try:
            res = minimize(lambda p: _smooth_score_radial(p, beta=25.0),
                          x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
            points = _extract_best_from_radial(res.x)
            try_update(points)
        except Exception:
            pass
        
        if time.time() > deadline - 85:
            break
        
        max_iter2 = min(2000, max(150, int(remaining * 0.08 / max(eval_time * _N, 1e-3))))
        try:
            res2 = minimize(lambda p: _score_radial(p),
                           res.x, method='Nelder-Mead',
                           options={'maxiter': max_iter2, 'xatol': 1e-12, 'fatol': 1e-12})
            points2 = _extract_best_from_radial(res2.x)
            try_update(points2)
        except Exception:
            pass
    
    # --- Phase 3: Joint (radii + angles) optimization ---
    joint_inits = []
    rng_c = np.random.default_rng(24680)
    for _ in range(4):
        small_angles = 0.03 * rng_c.standard_normal(_N)
        small_angles -= np.mean(small_angles)
        joint_inits.append(np.concatenate([np.ones(_N), small_angles]))
    ang_fourier = 0.05 * np.cos(2 * a)
    ang_fourier -= np.mean(ang_fourier)
    joint_inits.append(np.concatenate([np.ones(_N), ang_fourier]))
    joint_inits.append(np.concatenate([
        np.maximum(1.0 + 0.1 * np.cos(2 * a), 0.05),
        0.03 * np.cos(3 * a)
    ]))
    joint_inits.append(np.concatenate([
        np.maximum(1.0 + 0.15 * np.cos(4 * a), 0.05),
        0.04 * np.cos(2 * a)
    ]))
    
    for x0 in joint_inits:
        if time.time() > deadline - 60:
            break
        remaining = deadline - time.time()
        max_iter = min(1500, max(150, int(remaining * 0.08 / max(eval_time * 2 * _N, 1e-3))))
        try:
            res = minimize(lambda p: _smooth_score_joint(p, beta=25.0),
                          x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
            points = _extract_best_from_joint(res.x)
            try_update(points)
        except Exception:
            pass
        
        if time.time() > deadline - 45:
            break
        
        max_iter2 = min(1500, max(100, int(remaining * 0.06 / max(eval_time * 2 * _N, 1e-3))))
        try:
            res2 = minimize(lambda p: _score_joint(p),
                           res.x, method='Nelder-Mead',
                           options={'maxiter': max_iter2, 'xatol': 1e-12, 'fatol': 1e-12})
            points2 = _extract_best_from_joint(res2.x)
            try_update(points2)
        except Exception:
            pass
    
    # --- Phase 4: Polish best solution ---
    if best_points is not None and time.time() < deadline - 30:
        # Try to improve from current best using radial params
        # Extract approximate radii from best_points
        radii_est = np.linalg.norm(best_points, axis=1)
        angles_est = np.arctan2(best_points[:, 1], best_points[:, 0])
        
        # Polish with radial
        try:
            res = minimize(lambda p: _score_radial(p),
                          radii_est, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13})
            points = _extract_best_from_radial(res.x)
            try_update(points)
        except Exception:
            pass
        
        if time.time() < deadline - 20:
            # Polish with joint
            try:
                x_init = np.concatenate([radii_est, angles_est - _BASE_ANGLES])
                res = minimize(lambda p: _score_joint(p),
                              x_init, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13})
                points = _extract_best_from_joint(res.x)
                try_update(points)
            except Exception:
                pass
        
        if time.time() < deadline - 10:
            # Final high-precision polish
            try:
                radii_est2 = np.linalg.norm(best_points, axis=1)
                res = minimize(lambda p: _score_radial(p),
                              radii_est2, method='Powell',
                              options={'maxiter': 5000, 'xtol': 1e-14, 'ftol': 1e-14})
                points = _extract_best_from_radial(res.x)
                try_update(points)
            except Exception:
                pass
    
    # --- Finalize ---
    if best_points is None or best_score <= 0.0:
        # Fallback to regular polygon
        best_points = _normalize(_build_points(np.ones(_N)))
    
    return best_points
