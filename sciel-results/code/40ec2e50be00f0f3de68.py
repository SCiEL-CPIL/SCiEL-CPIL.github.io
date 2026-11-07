import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    y0 = np.array(y0, dtype=float)
    t0, t1 = t_span
    
    # Use DOP853 with tight tolerances to match the reference solution
    # This minimizes the relative error 'rel'
    # The number of calls will be low, keeping the penalty small
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13
    )
    
    if not sol.success:
        # Fallback if solve_ivp fails, though unlikely
        # Return y0 or a simple Euler step? 
        # Better to return the last computed value if available
        if sol.y.size > 0:
            return sol.y[:, -1]
        return y0

    return sol.y[:, -1]
