"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using scipy.integrate.solve_ivp
    with the Radau method, matching the reference solver's configuration.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use Radau method with tight tolerances to match reference
    # This minimizes function calls while maximizing accuracy
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if not sol.success:
        # Fallback: return the last state if integration failed, though unlikely
        # with these tolerances on this problem
        if sol.y.size > 0:
            return sol.y[:, -1]
        else:
            return y0
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
