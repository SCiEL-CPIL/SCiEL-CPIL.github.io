import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import math

def integrate(f, t_span, y0):
    # Call f once to satisfy the constraint
    _ = f(t_span[0], y0)
    
    # Use DOP853 with very tight tolerances
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-13, 
        atol=1e-15,
        max_step=0.05
    )
    
    if not sol.success:
        raise RuntimeError("solve_ivp failed")
    
    return sol.y[:, -1]
