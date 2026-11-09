"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit

    y0 = np.asarray(y0, dtype=float)
    t0, t1 = t_span

    # Use solve_ivp with DOP853 and very tight tolerances
    # The reference uses rtol=1e-11, atol=1e-13
    # We use slightly tighter tolerances to push rel below 1e-14
    # DOP853 is an 8th-order method, very accurate
    
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
            max_step=0.01  # limit step size for accuracy
        )
        
        if sol.success and sol.t[-1] == t1:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Fallback: if the above failed or didn't reach t1, try with even tighter settings
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16,
            max_step=0.005
        )
        
        if sol.success and sol.t[-1] == t1:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Last resort: high-step-count RK4
    steps = 10000
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    
    return y
# EVOLVE-BLOCK-END
