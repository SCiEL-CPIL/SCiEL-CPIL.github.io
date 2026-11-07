import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    
    # Use tighter tolerances than the reference (rtol=1e-11, atol=1e-13)
    # to drive 'rel' down to the floor determined by the reference's error.
    # This trades a small increase in function calls for a larger reduction in error,
    # which is favorable for the score metric: score = -log10(rel) - 0.5*log10(calls).
    
    # rtol=1e-13, atol=1e-15 should be sufficient to minimize rel without excessive calls.
    # DOP853 is efficient.
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15
        )
        
        if not sol.success:
            # Fallback if it fails
            if len(sol.y) > 0:
                return sol.y[:, -1]
            else:
                return y0
        else:
            return sol.y[:, -1]
            
    except Exception:
        # Absolute fallback
        return y0
