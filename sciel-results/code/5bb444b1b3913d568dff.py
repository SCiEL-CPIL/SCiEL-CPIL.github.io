import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    # Ensure y0 is a float array for compatibility
    y0 = np.array(y0, dtype=float)
    
    # Use BDF method. It is a different implicit method family than Radau.
    # BDF (Backward Differentiation Formula) is suitable for stiff problems.
    # It may have different efficiency characteristics (calls) and accuracy (rel)
    # compared to Radau.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='BDF', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if not sol.success:
        if sol.y.size > 0:
            return sol.y[:, -1]
        else:
            return y0

    return sol.y[:, -1]
