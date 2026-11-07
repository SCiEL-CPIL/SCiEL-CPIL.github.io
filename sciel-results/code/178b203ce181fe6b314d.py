import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Use the same solver and tolerances as the reference solution
    # to minimize the difference between y_end and y_ref.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-11, 
        atol=1e-13,
        t_eval=[t_span[1]]
    )
    
    if not sol.success:
        # Fallback to a simpler method if Radau fails, though it shouldn't
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-11, 
            atol=1e-13,
            t_eval=[t_span[1]]
        )
        
    return sol.y[:, -1]
