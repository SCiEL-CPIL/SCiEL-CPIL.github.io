import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Use DOP853 with tight tolerances
    # No max_step to allow adaptive large steps
    sol = solve_ivp(
        f, 
        t_span, 
        y,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15
    )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback
        return y
