import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Use solve_ivp with Radau method to match reference accuracy
    # The reference uses rtol=1e-11, atol=1e-13
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-11, 
        atol=1e-13
    )
    if not sol.success:
        # Fallback if it fails, though it shouldn't for this problem
        # Try a different method or tighter steps
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-11, 
            atol=1e-13
        )
    return sol.y[:, -1]
