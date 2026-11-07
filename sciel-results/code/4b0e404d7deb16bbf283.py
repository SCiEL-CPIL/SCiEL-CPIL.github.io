"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    """Integrate the harmonic oscillator using DOP853 with tight tolerances."""
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Use DOP853 with the same tolerances as the reference for high accuracy
    # This should give results very close to the reference with a reasonable call count
    sol = solve_ivp(
        f,
        t_span,
        y0_arr,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14,
        dense_output=False,
        max_step=(t1 - t0) / 50  # Limit step size for accuracy
    )
    
    if sol.success and sol.t[-1] == t1:
        return sol.y[:, -1]
    
    # Fallback: use RK4 with many steps if solve_ivp fails
    h = (t1 - t0) / 10000
    t, y = t0, y0_arr.copy()
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
