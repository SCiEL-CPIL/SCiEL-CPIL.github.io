import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    y0 = np.array(y0, dtype=float)
    t0, t1 = t_span
    
    # Use solve_ivp with tighter tolerances than reference to minimize rel error
    # Reference uses rtol=1e-11, atol=1e-13
    # We use rtol=1e-13, atol=1e-15 to get closer to the true solution
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-13, 
        atol=1e-15
    )
    
    if not sol.success:
        # Fallback to reference settings if tighter fails
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        
    return sol.y[:, -1]
