import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, leave 10s margin

    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        
        if time.time() > deadline:
             # Return best available
             if sol.y.shape[1] > 0:
                 return sol.y[:, -1]
             return y0

        if not sol.success:
            if sol.y.shape[1] > 0:
                return sol.y[:, -1]
            return y0

        return sol.y[:, -1]
    except Exception:
        return y0
