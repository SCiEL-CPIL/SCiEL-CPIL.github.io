# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Precompute sofa points
_SOFA_U = np.arange(-3.0, 1.5, 0.01)
_SOFA_V = np.arange(-1.5, 1.5, 0.01)
_SOFA_X, _SOFA_Y = np.meshgrid(_SOFA_U, _SOFA_V, indexing='ij')
_SOFA_PTS = np.stack([_SOFA_X.ravel(), _SOFA_Y.ravel()], axis=1)  # shape (N, 2)
_N_SOFA = _SOFA_PTS.shape[0]

# Deadline guard
_DEADLINE = 600.0
_START_TIME = None


def _elapsed():
    return time.time() - _START_TIME if _START_TIME is not None else 0.0


def _check_deadline():
    return _elapsed() > _DEADLINE - 30.0  # 30s safety margin


def _generate_placements(waypoints):
    P0 = np.array([0.0, 0.0, 0.0])
    PN = np.array([-2.0, 0.0, 90.0])
    full = [P0] + [np.array(p, dtype=np.float64) for p in waypoints] + [PN]
    placements = [full[0].copy()]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            placements.append(p1 + t * (p2 - p1))
    return np.array(placements, dtype=np.float64)


def _compute_sofa_mask(waypoints):
    placements = _generate_placements(waypoints)
    n_pl = len(placements)
    mask = np.ones(_N_SOFA, dtype=bool)
    for idx in range(n_pl):
        x, y, a_deg = placements[idx]
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        dx = _SOFA_PTS[:, 0] - x
        dy = _SOFA_PTS[:, 1] - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        inside = arm1 | arm2
        mask &= inside
        if not mask.any():
            break
    return mask


def _compute_area(waypoints):
    mask = _compute_sofa_mask(waypoints)
    return mask.sum() * 0.0001


def _soft_min_area(waypoints, beta=20.0):
    placements = _generate_placements(waypoints)
    n_pl = len(placements)
    min_margin = np.full(_N_SOFA, np.inf, dtype=np.float64)
    for idx in range(n_pl):
        x, y, a_deg = placements[idx]
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        dx = _SOFA_PTS[:, 0] - x
        dy = _SOFA_PTS[:, 1] - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        du1 = np.maximum(0.0, np.maximum(0.0, -u), np.maximum(0.0, u - 1.0))
        dv1 = np.maximum(0.0, np.maximum(0.0, -3.0 - v), np.maximum(0.0, v - 1.0))
        d1 = np.sqrt(du1 * du1 + dv1 * dv1)
        du2 = np.maximum(0.0, np.maximum(0.0, -3.0 - u), np.maximum(0.0, u - 1.0))
        dv2 = np.maximum(0.0, np.maximum(0.0, -v), np.maximum(0.0, v - 1.0))
        d2 = np.sqrt(du2 * du2 + dv2 * dv2)
        min_d = np.minimum(d1, d2)
        margin = -min_d
        min_margin = np.minimum(min_margin, margin)
        if np.all(min_margin < -1.0):
            break
    contrib = 1.0 / (1.0 + np.exp(-beta * min_margin))
    return contrib.sum() * 0.0001


def _score_neg(params):
    waypoints = _params_to_waypoints(params)
    return -_compute_area(waypoints)


def _score_neg_soft(params, beta=20.0):
    waypoints = _params_to_waypoints(params)
    return -_soft_min_area(waypoints, beta)


def _params_to_waypoints(params):
    n_wp = len(params) // 3
    wps = []
    for i in range(n_wp):
        wps.append([params[3*i], params[3*i+1], params[3*i+2]])
    return wps


def _waypoints_to_params(waypoints):
    params = []
    for wp in waypoints:
        params.extend([wp[0], wp[1], wp[2]])
    return np.array(params, dtype=np.float64)


def _initial_guesses():
    guesses = []
    baseline = [
        [-0.1, 0.1, 15.0], [-0.2, 0.2, 20.0], [-0.3, 0.3, 34.0], [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0], [-0.5, 0.2, 55.0], [-0.6, 0.2, 75.0], [-0.7, 0.3, 86.0],
    ]
    guesses.append(_waypoints_to_params(baseline))
    s1 = [
        [-0.1, 0.0, 10.0], [-0.2, 0.2, 25.0], [-0.3, 0.4, 40.0], [-0.4, 0.4, 50.0],
        [-0.5, 0.3, 60.0], [-0.6, 0.2, 70.0], [-0.7, 0.3, 80.0], [-0.75, 0.4, 86.0],
    ]
    guesses.append(_waypoints_to_params(s1))
    s2 = [
        [-0.05, 0.15, 20.0], [-0.15, 0.25, 35.0], [-0.25, 0.35, 45.0], [-0.35, 0.35, 55.0],
        [-0.45, 0.3, 65.0], [-0.55, 0.25, 75.0], [-0.65, 0.3, 82.0], [-0.7, 0.35, 87.0],
    ]
    guesses.append(_waypoints_to_params(s2))
    asym = [
        [-0.15, 0.05, 12.0], [-0.25, 0.15, 22.0], [-0.35, 0.25, 38.0], [-0.35, 0.35, 45.0],
        [-0.45, 0.4, 55.0], [-0.55, 0.3, 65.0], [-0.65, 0.2, 78.0], [-0.7, 0.3, 85.0],
    ]
    guesses.append(_waypoints_to_params(asym))
    late = [
        [-0.1, 0.0, 5.0], [-0.2, 0.1, 15.0], [-0.3, 0.2, 30.0], [-0.4, 0.3, 45.0],
        [-0.5, 0.3, 55.0], [-0.6, 0.3, 68.0], [-0.65, 0.35, 78.0], [-0.7, 0.4, 86.0],
    ]
    guesses.append(_waypoints_to_params(late))
    return guesses


def _try_optimize(x0, best_waypoints, best_area, maxiter_soft=400, maxiter_hard=200, beta=20.0):
    try:
        res = minimize(_score_neg_soft, x0, method='Nelder-Mead', args=(beta,),
                       options={'maxiter': maxiter_soft, 'xatol': 1e-4, 'fatol': 1e-6})
        if not _check_deadline():
            candidate_wp = _params_to_waypoints(res.x)
            cand_area = _compute_area(candidate_wp)
            if cand_area > best_area:
                best_area = cand_area
                best_waypoints = candidate_wp
    except Exception:
        pass
    if not _check_deadline():
        x0 = _waypoints_to_params(best_waypoints)
        try:
            res = minimize(_score_neg, x0, method='Nelder-Mead',
                           options={'maxiter': maxiter_hard, 'xatol': 1e-5, 'fatol': 1e-8})
            candidate_wp = _params_to_waypoints(res.x)
            cand_area = _compute_area(candidate_wp)
            if cand_area > best_area:
                best_area = cand_area
                best_waypoints = candidate_wp
        except Exception:
            pass
    return best_waypoints, best_area


def _extend_waypoints(waypoints, n_new):
    n_old = len(waypoints)
    new_wp = []
    for i in range(n_new):
        t = i / (n_new - 1) * (n_old - 1)
        idx = min(int(t), n_old - 2)
        frac = t - idx
        p1 = waypoints[idx]
        p2 = waypoints[idx + 1]
        new_wp.append([
            p1[0] + frac * (p2[0] - p1[0]),
            p1[1] + frac * (p2[1] - p1[1]),
            p1[2] + frac * (p2[2] - p1[2]),
        ])
    return new_wp


def moving_sofa_2d() -> np.ndarray:
    global _START_TIME
    _START_TIME = time.time()
    
    best_waypoints = [
        [-0.1, 0.1, 15.0], [-0.2, 0.2, 20.0], [-0.3, 0.3, 34.0], [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0], [-0.5, 0.2, 55.0], [-0.6, 0.2, 75.0], [-0.7, 0.3, 86.0],
    ]
    best_area = _compute_area(best_waypoints)
    
    guesses = _initial_guesses()
    for x0 in guesses:
        if _check_deadline():
            break
        best_waypoints, best_area = _try_optimize(
            x0, best_waypoints, best_area, maxiter_soft=300, maxiter_hard=150)
    
    if not _check_deadline():
        wp12 = _extend_waypoints(best_waypoints, 12)
        x0 = _waypoints_to_params(wp12)
        best_waypoints, best_area = _try_optimize(
            x0, best_waypoints, best_area, maxiter_soft=300, maxiter_hard=150)
    
    if not _check_deadline():
        wp16 = _extend_waypoints(best_waypoints, 16)
        x0 = _waypoints_to_params(wp16)
        best_waypoints, best_area = _try_optimize(
            x0, best_waypoints, best_area, maxiter_soft=250, maxiter_hard=120)
    
    if not _check_deadline():
        x0 = _waypoints_to_params(best_waypoints)
        try:
            res = minimize(_score_neg, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-9})
            candidate_wp = _params_to_waypoints(res.x)
            cand_area = _compute_area(candidate_wp)
            if cand_area > best_area:
                best_area = cand_area
                best_waypoints = candidate_wp
        except Exception:
            pass
    
    result = np.array(best_waypoints, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        result = np.array([
            [-0.1, 0.1, 15.0], [-0.2, 0.2, 20.0], [-0.3, 0.3, 34.0], [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0], [-0.5, 0.2, 55.0], [-0.6, 0.2, 75.0], [-0.7, 0.3, 86.0],
        ], dtype=np.float64)
    return result

# EVOLVE-BLOCK-END
