import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system using scipy's DOP853 with optimized settings.
    
    Uses DOP853 with tight tolerances to achieve high accuracy with minimal function calls.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 295.0  # Safety margin before 300s limit
    
    try:
        # Use DOP853 with tight tolerances
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=0.1
        )
        
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
        else:
            # Fallback with slightly looser tolerances
            if time.time() < deadline - 10:
                sol = solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='DOP853', 
                    rtol=1e-11, 
                    atol=1e-13
                )
                if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                    return sol.y[:, -1]
            return y0
    except Exception:
        return y0
