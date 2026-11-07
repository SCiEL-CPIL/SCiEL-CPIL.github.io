"""Improved Van der Pol integrator with tighter tolerances."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using Radau with very tight tolerances
    to minimize relative error, accepting a modest increase in function calls.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    deadline = time.time() + 280.0
    
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-13,
            atol=1e-15
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: RK4
    steps = 100000
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        if time.time() > deadline - 5.0:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
