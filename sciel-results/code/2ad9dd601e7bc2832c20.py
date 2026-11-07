import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # LSODA is an adaptive multi-step method that switches between
    # Adams (non-stiff) and BDF (stiff). It is a different algorithmic
    # family from the explicit Runge-Kutta methods used in the seed.
    # We use tight tolerances to ensure accuracy.
    sol = solve_ivp(f, t_span, y0, method='LSODA', rtol=1e-11, atol=1e-13)
    if not sol.success:
        # Fallback to Radau if LSODA fails
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-11, atol=1e-13)
        if not sol.success:
            raise RuntimeError("Integration failed")
            
    return sol.y[:, -1]
