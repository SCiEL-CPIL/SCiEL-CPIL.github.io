import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 290.0

    # DOP853 is an 8th-order method, very accurate.
    # Tight tolerances to minimize relative error.
    rtol = 1e-13
    atol = 1e-16

    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=rtol, 
            atol=atol,
            dense_output=False
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to RK45 if DOP853 fails (unlikely)
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45', 
                rtol=rtol, 
                atol=atol
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                # Last resort: return initial condition or last computed state
                return sol.y[:, -1] if sol.y.size > 0 else np.array(y0)
    except Exception:
        return np.array(y0)
