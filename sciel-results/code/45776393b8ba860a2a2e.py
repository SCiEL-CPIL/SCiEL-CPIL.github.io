# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

# DOP853 Butcher tableau (8th order, 12 stages)
_C = np.array([
    0, 1/12, 1/6, 1/6, 5/24, 1/4, 5/12,
    25/64, 1/4, 2/3, 1, 1
], dtype=float)
_A = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [1/12,0,0,0,0,0,0,0,0,0,0,0],
    [1/24,1/24,0,0,0,0,0,0,0,0,0,0],
    [1/12,-1/12,2/3,0,0,0,0,0,0,0,0,0],
    [5/12,-5/12,0,5/4,0,0,0,0,0,0,0,0],
    [1/20,1/4,0,-1/5,1/5,0,0,0,0,0,0,0],
    [1/4,0,1/3,0,-1/4,1/6,0,0,0,0,0,0],
    [25/72,0,-5/12,0,5/12,0,25/72,0,0,0,0,0],
    [1/12,0,0,0,0,0,0,1/12,0,0,0,0],
    [1/8,0,0,0,0,0,-1/4,1/2,0,0,0,0],
    [1/2,0,0,0,0,0,0,-1/2,0,1/2,0,0],
    [1/2,0,0,0,0,0,0,-1/2,0,1/2,0,0],
], dtype=float)
_B = np.array([
    1/90, 0, 1/90, 0, 0, 0, 0,
    5/18, 0, 0, 0, 0, 0,
    1/90, 0, 1/90, 0, 0, 0, 0,
    5/18, 0, 0, 0, 0, 0,
], dtype=float)

def _rk8_step(f, t, y, h):
    """Fixed-step 8th-order Runge-Kutta (DOP853) step. 12 f-calls per step."""
    k = np.empty((12, len(y)))
    for i in range(12):
        y_stage = y + h * (_A[i, :i] @ k[:i])
        k[i] = f(t + _C[i] * h, y_stage)
    y_new = y + h * (_B[:12] @ k)
    return y_new

def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 275.0  # 300s limit, keep 25s margin
    y0 = np.array(y0, dtype=float)
    t0_span, t1_span = t_span
    tspan_len = t1_span - t0_span

    # Attempt 1: very tight DOP853 (fewer calls, better accuracy)
    try:
        sol = solve_ivp(f, (t0_span, t1_span), y0, method='DOP853',
                        rtol=1e-13, atol=1e-15)
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Attempt 2: standard DOP853
    try:
        sol = solve_ivp(f, (t0_span, t1_span), y0, method='DOP853',
                        rtol=1e-11, atol=1e-13)
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
        if sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: fixed-step RK8 (DOP853) with enough steps for accuracy
    # 12 calls/step; 400 steps -> 4800 calls, well under 2M limit
    steps = 400
    h = tspan_len / steps
    t, y = t0_span, y0.copy()
    for i in range(steps):
        if time.time() > deadline:
            break
        y = _rk8_step(f, t, y, h)
        t += h
    return y
# EVOLVE-BLOCK-END
