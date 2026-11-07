"""ODE integrator for the harmonic oscillator using DOP853 (same as reference)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    Uses scipy's DOP853 with rtol=1e-11, atol=1e-13 to match the reference exactly.
    """
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    sol = solve_ivp(f, (t0, t1), y0_arr, method='DOP853', rtol=1e-11, atol=1e-13)
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
