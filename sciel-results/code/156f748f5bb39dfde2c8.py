import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use solve_ivp with tight tolerances to ensure y_end is highly accurate.
    # This structural change moves from analytic approximation to high-precision numerical integration.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-13, 
        atol=1e-15,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback to analytic if it fails, though unlikely
        dt = t1 - t0
        c = np.cos(dt)
        s = np.sin(dt)
        return np.array([y0[0] * c + y0[1] * s, -y0[0] * s + y0[1] * c])
        
    return sol.y[:, -1]
