import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    t0, t1 = t_span
    sol = scipy.integrate.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    if not sol.success:
        # Fallback if DOP853 fails (unlikely)
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13
        )
    return sol.y[:, -1]
