# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Precompute sofa points
_SOFA_U = np.arange(-3.0, 1.5, 0.01)
_SOFA_V = np.arange(-1.5, 1.5, 0.01)
_SOFA_X, _SOFA_Y = np.meshgrid(_SOFA_U, _SOFA_V, indexing='ij')
_N_SOFa = _SOFA_X.size
_SOFA_X_1D = _SOFA_X.ravel()
_SOFA_Y_1D = _SOFA_Y.ravel()


def _generate_placements(waypoints):
    """Generate all placements including start, waypoints, end, with linear interpolation."""
    P0 = np.array([0.0, 0.0, 0.0])
    PN = np.array([-2.0, 0.0, 90.0])
    full = [P0] + [np.asarray(p, dtype=np.float64) for p in waypoints] + [PN]
    placements = [full[0].copy()]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da - 1e-9)))
        for k in range(1, m + 1):
            t = k / m
            placements.append(p1 + t * (p2 - p1))
    return placements


def _compute_area(waypoints):
    """Compute the area of the sofa intersection. Returns float."""
    placements = _generate_placements(waypoints)
    inside_all = np.ones(_N_SOFa, dtype=bool)
    for (x, y, a_deg) in placements:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        dx = _SOFA_X_1D - x
        dy = _SOFA_Y_1D - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        in_h = (u >= -1e-12) & (u <= 1 + 1e-12) & (v >= -3 - 1e-12) & (v <= 1 + 1e-12)
        in_v = (u >= -3 - 1e-12) & (u <= 1 + 1e-12) & (v >= -1e-12) & (v <= 1 + 1e-12)
        in_base = in_h | in_v
        inside_all &= in_base
        if not inside_all.any():
            break
    return inside_all.sum() * 0.0001


def _neg_area(params):
    n_wp = params.shape[0] // 3
    waypoints = params.reshape(n_wp, 3)
    return -_compute_area(waypoints)


def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # Safety margin before 630s kill

    N_WAYPOINTS = 20  # Increased waypoints for smoother path representation
    
    # Strategy: Evaluate a few strong analytic paths first.
    # Knowledge suggests semi-circle or smooth S-curves are effective.
    
    candidates = []
    
    ts = np.linspace(0, 1, N_WAYPOINTS)
    idx = np.arange(1, N_WAYPOINTS)  # Exclude endpoints as evaluator adds them
    
    # Candidate 1: Standard S-curve with various amplitudes
    for amp in [0.2, 0.3, 0.4, 0.5]:
        for x_scale in [1.0, 1.1, 1.2]:
            x = -x_scale * 2.0 * ts
            y = amp * np.sin(np.pi * ts)
            a = 90.0 * ts
            wp = np.column_stack([x[idx], y[idx], a[idx]])
            candidates.append(wp)
            
    # Candidate 2: Power-law S-curve
    for p in [0.8, 1.0, 1.2]:
        for amp in [0.3, 0.4]:
            x = -2.0 * (ts**p)
            y = amp * np.sin(np.pi * ts**p)
            a = 90.0 * ts
            wp = np.column_stack([x[idx], y[idx], a[idx]])
            candidates.append(wp)

    # Evaluate candidates
    best_wp = None
    best_area = -1.0
    
    for wp in candidates:
        if time.time() > deadline - 60:
            break
        try:
            area = _compute_area(wp)
            if area > best_area:
                best_area = area
                best_wp = wp
        except Exception:
            continue
            
    if best_wp is None:
        # Fallback to a simple linear path
        best_wp = np.column_stack([
            -2.0 * ts[idx],
            0.0 * ts[idx],
            90.0 * ts[idx]
        ])
        best_area = _compute_area(best_wp)

    # If time remains, attempt a local polish on the best candidate
    if time.time() < deadline - 30:
        try:
            params = best_wp.ravel().copy()
            
            result = minimize(
                _neg_area,
                params,
                method='Nelder-Mead',
                options={
                    'maxiter': 50,  # Very low to prevent timeout
                    'maxfev': 50,
                    'xatol': 1e-4,
                    'fatol': 1e-5,
                }
            )
            
            if result.fun < -best_area:
                new_wp = result.x.reshape(N_WAYPOINTS, 3)
                if np.all(np.isfinite(new_wp)):
                    new_area = -result.fun
                    if new_area > best_area:
                        best_wp = new_wp
                        best_area = new_area
        except Exception:
            pass

    # Final safety check
    if not np.all(np.isfinite(best_wp)):
        best_wp = np.column_stack([
            -2.0 * ts[idx],
            0.0 * ts[idx],
            90.0 * ts[idx]
        ])

    return best_wp

# EVOLVE-BLOCK-END
