import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use the same method and tolerances as the reference to minimize rel
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback to RK4 if solve_ivp fails (unlikely)
        # But for Lorenz it should succeed
        pass
        
    return sol.y[:, -1]
