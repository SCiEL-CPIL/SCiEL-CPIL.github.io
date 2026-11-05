import time
import math
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 145.0

def _log_score(params):
    """Compute log of Delta/n^n from 16 coordinate params. Returns -inf if invalid."""
    pts = params.reshape(8, 2)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    off_diag = dists[~np.eye(8, dtype=bool)]
    if np.any(off_diag < 1e-10):
        return -np.inf
    log_prod = np.sum(np.log(dists))
    max_d = np.max(dists)
    if max_d < 1e-15:
        return -np.inf
    n = 8
    log_prod_scaled = log_prod + n * (n - 1) * math.log(2.0 / max_d)
    log_score = log_prod_scaled - n * math.log(n)
    return log_score

def _score(params):
    ls = _log_score(params)
    if ls < -1e6:
        return 0.0
    return math.exp(ls)

def _regular_octagon():
    pts = []
    for k in range(8):
        angle = 2 * math.pi * k / 8
        pts.append([math.cos(angle), math.sin(angle)])
    return np.array(pts, dtype=np.float64)

def _two_concentric_squares(r1, r2, rot1, rot2):
    pts = []
    for k in range(4):
        angle = rot1 + 2 * math.pi * k / 4
        pts.append([r1 * math.cos(angle), r1 * math.sin(angle)])
    for k in range(4):
        angle = rot2 + 2 * math.pi * k / 4
        pts.append([r2 * math.cos(angle), r2 * math.sin(angle)])
    return np.array(pts, dtype=np.float64)

def _random_initial(spread=0.5):
    angles = 2 * math.pi * np.arange(8) / 8 + np.random.randn(8) * spread
    radii = 1.0 + np.random.randn(8) * 0.3
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _neg_log_score(params):
    if time.time() > _DEADLINE:
        return 1e10
    ls = _log_score(params)
    if ls < -1e6:
        return 1e10
    return -ls

def _neg_smooth_log_score(params):
    """Smooth surrogate: uses soft-max for diameter instead of hard max."""
    if time.time() > _DEADLINE:
        return 1e10
    pts = params.reshape(8, 2)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diffs**2, axis=2)
    np.fill_diagonal(dists2, 0.0)
    dists = np.sqrt(dists2)
    off_diag = dists[~np.eye(8, dtype=bool)]
    if np.any(off_diag < 1e-10):
        return 1e10
    log_prod = np.sum(np.log(dists))
    # Soft-max of distances (temperature beta)
    beta = 50.0
    max_d = np.max(dists)
    soft_max = max_d + math.log(1.0) / beta + np.log(np.sum(np.exp((dists - max_d) * beta))) / beta
    if soft_max < 1e-15:
        return 1e10
    n = 8
    log_prod_scaled = log_prod + n * (n - 1) * math.log(2.0 / soft_max)
    log_score = log_prod_scaled - n * math.log(n)
    if log_score < -1e6:
        return 1e10
    return -log_score

# ===== Complex polynomial approach =====

def _poly_to_coefficients(pts):
    """Given 8 complex points, compute coefficients of P(z) = prod(z - z_i) = z^8 + c7 z^7 + ... + c0.
    Returns array [c0, c1, ..., c7] (real and imag interleaved for optimization).
    Actually returns 14 real values: Re(c0), Im(c0), Re(c1), Im(c1), ..., Re(c7), Im(c7)"""
    z = pts[:, 0] + 1j * pts[:, 1]
    # Compute coefficients using Newton's identities or direct polynomial multiplication
    coeffs = [1 + 0j]  # start with 1
    for zi in z:
        # multiply by (z - zi)
        new_coeffs = [0j] * (len(coeffs) + 1)
        for i in range(len(coeffs)):
            new_coeffs[i] += coeffs[i] * (-zi)
            new_coeffs[i + 1] += coeffs[i]
        coeffs = new_coeffs
    # coeffs[0] = c0, coeffs[1] = c1, ..., coeffs[8] = 1 (leading)
    # We optimize c0..c7 (14 real params)
    params = np.zeros(14)
    for k in range(8):
        params[2 * k] = coeffs[k].real
        params[2 * k + 1] = coeffs[k].imag
    return params

def _coefficients_to_points(coeffs_14):
    """Given 14 real params [Re(c0), Im(c0), ..., Re(c7), Im(c7)], find roots of P(z)."""
    coeffs = []
    for k in range(8):
        coeffs.append(complex(coeffs_14[2 * k], coeffs_14[2 * k + 1]))
    # P(z) = z^8 + c7*z^7 + ... + c0
    # numpy.roots expects highest degree first
    poly_coeffs = [1.0 + 0j] + list(reversed(coeffs))
    roots = np.roots(poly_coeffs)
    pts = np.column_stack([roots.real, roots.imag])
    return pts

def _log_discriminant_from_coeffs(coeffs_14):
    """Compute log|Disc(P)| where P has coefficients c0..c7 (monic of degree 8).
    Disc = prod_{i<j} (z_i - z_j)^2. So log|Disc| = 2 * sum_{i<j} log|z_i - z_j|."""
    pts = _coefficients_to_points(coeffs_14)
    # Check for distinctness
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    off_diag = dists[~np.eye(8, dtype=bool)]
    if np.any(off_diag < 1e-12):
        return -np.inf
    log_prod = np.sum(np.log(dists))
    return log_prod  # this is sum_{i<j} log|zi-zj|, not doubled

def _neg_poly_objective(params_14, beta=50.0):
    """Objective for polynomial coefficient optimization.
    Maximizes log-discriminant subject to soft-max constraint on root moduli.
    The diameter constraint: max pairwise distance <= 2 (after scaling).
    We use: penalty = beta * (soft_max_diameter - 2)^2 if soft_max_diameter > 2."""
    if time.time() > _DEADLINE:
        return 1e10
    try:
        pts = _coefficients_to_points(params_14)
        # Check validity
        if not np.all(np.isfinite(pts)):
            return 1e10
        # Check distinctness
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists2 = np.sum(diffs**2, axis=2)
        np.fill_diagonal(dists2, 1.0)
        min_d2 = np.min(dists2)
        if min_d2 < 1e-20:
            return 1e10
        dists = np.sqrt(dists2)
        # Log product of all pairwise distances (ordered pairs: n*(n-1) terms)
        log_prod = np.sum(np.log(dists))  # this is sum over all ordered pairs
        # Soft-max diameter
        max_d = np.max(dists)
        soft_max_d = max_d + np.log(np.sum(np.exp((dists - max_d) * beta))) / beta
        if soft_max_d < 1e-15:
            return 1e10
        # Scale to diameter 2
        n = 8
        log_prod_scaled = log_prod + n * (n - 1) * math.log(2.0 / soft_max_d)
        log_score = log_prod_scaled - n * math.log(n)
        if log_score < -1e6:
            return 1e10
        return -log_score
    except Exception:
        return 1e10

def _neg_poly_objective_v2(params_14, beta=100.0):
    """Alternative: directly optimize log-discriminant with soft penalty on max root modulus.
    Constraint: all |z_i| <= R where R is chosen so diameter = 2.
    For 8 points, if we constrain max|z_i| = 1, diameter <= 2.
    Penalty: beta * sum_i max(0, |z_i|^2 - 1)^2"""
    if time.time() > _DEADLINE:
        return 1e10
    try:
        pts = _coefficients_to_points(params_14)
        if not np.all(np.isfinite(pts)):
            return 1e10
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists2 = np.sum(diffs**2, axis=2)
        np.fill_diagonal(dists2, 1.0)
        min_d2 = np.min(dists2)
        if min_d2 < 1e-20:
            return 1e10
        dists = np.sqrt(dists2)
        log_prod = np.sum(np.log(dists))
        max_d = np.max(dists)
        if max_d < 1e-15:
            return 1e10
        n = 8
        log_prod_scaled = log_prod + n * (n - 1) * math.log(2.0 / max_d)
        log_score = log_prod_scaled - n * math.log(n)
        if log_score < -1e6:
            return 1e10
        return -log_score
    except Exception:
        return 1e10

def _optimize_poly(params_14, method='L-BFGS-B', maxiter=200, beta=50.0):
    """Optimize polynomial coefficients."""
    try:
        if method == 'L-BFGS-B':
            result = minimize(
                _neg_poly_objective, params_14, args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
        elif method == 'Nelder-Mead':
            result = minimize(
                _neg_poly_objective, params_14, args=(beta,),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
        elif method == 'Powell':
            result = minimize(
                _neg_poly_objective, params_14, args=(beta,),
                method='Powell',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'xtol': 1e-8}
            )
        else:
            return None
        return result
    except Exception:
        return None

def _optimize_coords(params, method='L-BFGS-B', maxiter=200):
    """Optimize coordinates directly."""
    try:
        if method == 'L-BFGS-B':
            result = minimize(
                _neg_log_score, params,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
        elif method == 'Nelder-Mead':
            result = minimize(
                _neg_log_score, params,
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
        elif method == 'Powell':
            result = minimize(
                _neg_log_score, params,
                method='Powell',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'xtol': 1e-9}
            )
        else:
            return None
        return result
    except Exception:
        return None

def _antipodal_initial():
    """Antipodal pair initialization: 4 pairs of antipodal points on unit circle."""
    pts = []
    for k in range(4):
        angle = 2 * math.pi * k / 4 + 0.1
        pts.append([math.cos(angle), math.sin(angle)])
        pts.append([-math.cos(angle), -math.sin(angle)])
    return np.array(pts, dtype=np.float64)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 145.0

    np.random.seed(42)

    best_params = _regular_octagon().flatten()
    best_log_score = _log_score(best_params)

    def _try_update(params_candidate):
        nonlocal best_params, best_log_score
        if params_candidate is not None:
            ls = _log_score(params_candidate)
            if ls > best_log_score:
                best_log_score = ls
                best_params = params_candidate.copy()

    # Phase 1: Coordinate-space optimization from diverse starts
    initial_configs = [_regular_octagon(), _antipodal_initial()]
    for r1 in [0.3, 0.5, 0.7]:
        for r2 in [1.0, 1.3]:
            for rot1 in [0.0, math.pi / 8]:
                initial_configs.append(_two_concentric_squares(r1, r2, rot1, rot1 + math.pi / 4))
    for _ in range(8):
        initial_configs.append(_random_initial(spread=np.random.uniform(0.2, 0.6)))

    for init_pts in initial_configs:
        if time.time() > _DEADLINE - 10:
            break
        params = init_pts.flatten()
        # Nelder-Mead
        result = _optimize_coords(params, method='Nelder-Mead', maxiter=300)
        _try_update(result.x if result is not None else None)
        # L-BFGS-B polish
        if time.time() > _DEADLINE - 10:
            break
        result2 = _optimize_coords(best_params, method='L-BFGS-B', maxiter=100)
        _try_update(result2.x if result2 is not None else None)

    # Phase 2: Polynomial coefficient optimization
    # Convert best to poly coefficients
    best_pts = best_params.reshape(8, 2)
    poly_params = _poly_to_coefficients(best_pts)

    # Optimize in poly space
    for beta in [50.0, 100.0, 200.0]:
        if time.time() > _DEADLINE - 15:
            break
        result = _optimize_poly(poly_params, method='L-BFGS-B', maxiter=300, beta=beta)
        if result is not None and result.fun < 1e8:
            # Convert back to coordinates
            pts_from_poly = _coefficients_to_points(result.x)
            if np.all(np.isfinite(pts_from_poly)):
                coord_params = pts_from_poly.flatten()
                _try_update(coord_params)
                # Update poly params for next iteration
                if np.all(np.isfinite(coord_params)):
                    poly_params = _poly_to_coefficients(pts_from_poly)

    # Phase 3: Multi-beta polishing chain in coordinate space
    for beta in [20.0, 50.0, 100.0, 200.0, 500.0]:
        if time.time() > _DEADLINE - 10:
            break
        # Use smooth objective with this beta
        def _smooth_obj(p, b=beta):
            if time.time() > _DEADLINE:
                return 1e10
            pts = p.reshape(8, 2)
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists2 = np.sum(diffs**2, axis=2)
            np.fill_diagonal(dists2, 1.0)
            min_d2 = np.min(dists2)
            if min_d2 < 1e-20:
                return 1e10
            dists = np.sqrt(dists2)
            log_prod = np.sum(np.log(dists))
            max_d = np.max(dists)
            if max_d < 1e-15:
                return 1e10
            soft_max = max_d + np.log(np.sum(np.exp((dists - max_d) * b))) / b
            n = 8
            log_prod_scaled = log_prod + n * (n - 1) * math.log(2.0 / soft_max)
            log_score = log_prod_scaled - n * math.log(n)
            if log_score < -1e6:
                return 1e10
            return -log_score

        try:
            result = minimize(
                _smooth_obj, best_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result is not None and result.fun < 1e8:
                _try_update(result.x)
        except Exception:
            pass

    # Phase 4: Final high-precision coordinate polish
    if time.time() < _DEADLINE - 5:
        result = _optimize_coords(best_params, method='L-BFGS-B', maxiter=500)
        _try_update(result.x if result is not None else None)
        if time.time() < _DEADLINE - 3:
            result = _optimize_coords(best_params, method='Powell', maxiter=100)
            _try_update(result.x if result is not None else None)

    # Phase 5: Random restarts in poly space
    np.random.seed(123)
    for _ in range(10):
        if time.time() > _DEADLINE - 5:
            break
        # Random poly coefficients (small perturbations of best)
        poly_noise = np.random.randn(14) * 0.1
        poly_try = poly_params + poly_noise
        result = _optimize_poly(poly_try, method='Nelder-Mead', maxiter=150, beta=100.0)
        if result is not None and result.fun < 1e8:
            pts_from_poly = _coefficients_to_points(result.x)
            if np.all(np.isfinite(pts_from_poly)):
                _try_update(pts_from_poly.flatten())

    # Final: ensure we have a valid answer
    final_pts = best_params.reshape(8, 2)
    result_list = []
    for p in final_pts:
        x = float(max(-1e6, min(1e6, p[0])))
        y = float(max(-1e6, min(1e6, p[1])))
        result_list.append([x, y])

    return result_list
