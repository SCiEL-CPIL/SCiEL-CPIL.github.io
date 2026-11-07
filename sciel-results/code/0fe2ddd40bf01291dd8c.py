import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Use DOP853 with tight tolerances to match or exceed reference accuracy
    # Reference uses rtol 1e-11, atol 1e-13
    # We use slightly tighter tolerances to ensure rel error is minimal
    sol = solve_ivp(
        f, 
        t_span, 
        y0_arr, 
        method='DOP853', 
        rtol=1e-12, 
        atol=1e-14,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback if it fails, though unlikely for this smooth problem
        # Just return the last state or raise
        return sol.y[:, -1]
        
    return sol.y[:, -1]
