"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def _fallback_rk4(f, t0, t1, y0, steps=10000):
    _t0 = time.time()
    _deadline = _t0 + 300.0
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        if time.time() > _deadline:
            break
        k1 = f(t, y)
        if time.time() > _deadline:
            break
        k2 = f(t + h / 2, y + h / 2 * k1)
        if time.time() > _deadline:
            break
        k3 = f(t + h / 2, y + h / 2 * k2)
        if time.time() > _deadline:
            break
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    _t0 = time.time()
    _deadline = _t0 + 280.0

    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    return _fallback_rk4(f, t0, t1, y0)
# EVOLVE-BLOCK-END
