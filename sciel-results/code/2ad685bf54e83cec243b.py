"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    y0 = np.array(y0, dtype=float)
    t0, t1 = t_span
    
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    sol = solve_ivp(f_counted, (float(t0), float(t1)), y0, 
                    method='Radau', rtol=1e-11, atol=1e-13, max_step=0.1)
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
