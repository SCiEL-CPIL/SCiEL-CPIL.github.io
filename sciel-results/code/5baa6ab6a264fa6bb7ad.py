"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    sol = solve_ivp(
        f_counted,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-13,
        atol=1e-14,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback: return last state if available
        if sol.y is not None and len(sol.y) > 0:
            return sol.y[:, -1]
        raise RuntimeError("Integration failed: " + str(sol.message))
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
