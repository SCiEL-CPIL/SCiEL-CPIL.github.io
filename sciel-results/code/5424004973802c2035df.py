"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the ODE system y' = f(t, y) over t_span starting from y0.
    Uses scipy's DOP853 with tight tolerances to match the reference generator.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: kill after 330s wall-clock, use 320s as safety margin
    deadline = time.time() + 320.0
    
    try:
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            dense_output=False
        )
        
        if sol.success and sol.t[-1] >= t1 - 1e-12:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
        
        # If solve_ivp failed, fall back to a high-accuracy RK4
        # This should rarely happen for the harmonic oscillator
        return _fallback_rk4(f, t0, t1, y0)
    
    except Exception:
        # On any exception, try the fallback
        return _fallback_rk4(f, t0, t1, y0)


def _fallback_rk4(f, t0, t1, y0, steps=5000):
    """High-step-count RK4 as a fallback."""
    h = (t1 - t0) / steps
    t = t0
    y = np.array(y0, dtype=float)
    
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
