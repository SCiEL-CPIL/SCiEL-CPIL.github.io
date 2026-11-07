"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def _fallback_rk4(f, t0, t1, y0, steps=10000):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    start_time = time.time()
    deadline = start_time + 250.0  # 50s margin
    
    try:
        # Use tight tolerances to minimize rel error
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback with deadline check
    # Estimate time per step. If we are close to deadline, reduce steps?
    # For now, just run. 10000 steps is fast.
    return _fallback_rk4(f, t0, t1, y0)
# EVOLVE-BLOCK-END
