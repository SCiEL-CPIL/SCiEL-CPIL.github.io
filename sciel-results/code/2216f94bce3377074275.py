"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Integrate using the provided function f with high precision
    # Using the exact method and tolerances specified in the task contract
    sol = solve_ivp(f, t_span, y0_arr, method='DOP853', rtol=1e-11, atol=1e-13)
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback: return initial state if integration fails (should not happen)
        return y0_arr
# EVOLVE-BLOCK-END
