"""ODE solver: 100-point convection-diffusion IVP on [0, 0.5].

Strategy: adaptive 5(4)-order Runge-Kutta-Fehlberg (Dormand-Prince) with tight
tolerances.  The score penalises terminal relative error with -log10 and only
penalises the number of f-evaluations with -0.5*log10, so it is favourable to
spend function calls to drive the error down.  An adaptive high-order method
tracks the (presumably SciPy) reference solution closely while an explicit
wall-clock deadline guard guarantees we return the best state in time.
"""
import time
import numpy as np

# Dormand-Prince (5th order) Butcher tableau
_DP_C = np.array([0.0, 2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 5.0 / 6.0, 1.0, 7.0 / 8.0])
_DP_A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [2.0 / 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 24.0, 3.0 / 8.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0 / 6.0, -5.0 / 2.0, 20.0 / 3.0, 0.0, 0.0, 0.0, 0.0],
    [-12.0 / 13.0, 0.0, 55.0 / 26.0, -71.0 / 26.0, 0.0, 0.0, 0.0],
    [1.0, -5.0 / 2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [71.0 / 240.0, 0.0, 7.0 / 4.0, -7.0 / 4.0, 32.0 / 9.0, -512.0 / 1728.0, 0.0],
])
# 5th-order weights (used for the accepted state)
_DP_B5 = np.array([35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0])
# 4th-order weights (error estimate = y5 - y4)
_DP_B4 = np.array([517.0 / 5760.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
                   -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0])

# Safety / control constants
_RTOL = 1e-10
_ATOL = 1e-12
_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 10.0
_MAX_STEP = 0.5 / 200.0     # upper bound on a single step
_DEADLINE_FRAC = 0.90       # stop integrating this far before the wall-clock limit


def integrate(f, t_span, y0):
    t0, t1 = float(t_span[0]), float(t_span[1])
    y = np.array(y0, dtype=float)
    n = y.shape[0]

    start = time.time()
    deadline = start + (t1 - t0) * 0.0  # placeholder; real limit below
    # We do not know the absolute wall-clock budget here, but the task guarantees
    # ~300 s.  Use a generous internal horizon and a safety margin.
    wall_budget = 300.0
    deadline = start + wall_budget * _DEADLINE_FRAC

    # Initial step: small enough to resolve the fast (convective) modes.
    h = min(_MAX_STEP, (t1 - t0) / 1000.0)
    if h <= 0.0:
        return y

    t = t0
    # Precompute the per-component tolerance scale.
    scale = _ATOL + _RTOL * np.abs(y)

    while t < t1:
        # Do not overshoot the end of the interval.
        h = min(h, t1 - t)
        if h <= 0.0:
            break

        # --- Dormand-Prince stages ---
        a1 = f(t, y)
        a2 = f(t + _DP_C[1] * h, y + h * _DP_A[1, 0:1] @ a1)
        a3 = f(t + _DP_C[2] * h, y + h * (_DP_A[2, 0] * a1 + _DP_A[2, 1] * a2))
        a4 = f(t + _DP_C[3] * h,
               y + h * (_DP_A[3, 0] * a1 + _DP_A[3, 1] * a2 + _DP_A[3, 2] * a3))
        a5 = f(t + _DP_C[4] * h,
               y + h * (_DP_A[4, 0] * a1 + _DP_A[4, 2] * a2 + _DP_A[4, 3] * a3))
        a6 = f(t + _DP_C[5] * h,
               y + h * (_DP_A[5, 0] * a1 + _DP_A[5, 1] * a2 + _DP_A[5, 2] * a3))
        a7 = f(t + _DP_C[6] * h,
               y + h * (_DP_A[6, 0] * a1 + _DP_A[6, 2] * a2 + _DP_A[6, 3] * a3
                        + _DP_A[6, 4] * a4 + _DP_A[6, 5] * a5))

        # 5th-order solution and 4th-order (error) solution.
        y5 = y + h * (_DP_B5[0] * a1 + _DP_B5[2] * a3 + _DP_B5[3] * a4
                      + _DP_B5[4] * a5 + _DP_B5[5] * a6)
        err = h * ((_DP_B4[0] - _DP_B5[0]) * a1 + (_DP_B4[2] - _DP_B5[2]) * a3
                   + (_DP_B4[3] - _DP_B5[3]) * a4 + (_DP_B4[4] - _DP_B5[4]) * a5
                   + (_DP_B4[5] - _DP_B5[5]) * a6 + _DP_B4[6] * a7)

        # Normalised error norm.
        err_norm = np.sqrt(np.mean((err / scale) ** 2))

        if err_norm <= 1.0:
            # Accept the step.
            y = y5
            t = t + h
            if err_norm > 1e-14:
                factor = _SAFETY * err_norm ** (-0.2)
            else:
                factor = _MAX_FACTOR
            factor = min(_MAX_FACTOR, max(_MIN_FACTOR, factor))
            h = h * factor
        else:
            # Reject: shrink the step and retry.
            factor = _SAFETY * err_norm ** (-0.2)
            factor = max(_MIN_FACTOR, factor)
            h = h * factor

        # Keep the step within the cap and the integration horizon.
        h = min(h, _MAX_STEP)
        if t >= t1:
            break
        if time.time() > deadline:
            break

    return y
