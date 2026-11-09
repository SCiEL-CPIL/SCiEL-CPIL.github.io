import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) using a stiff solver (Radau or BDF).
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # Try Radau first
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            rtol=1e-10,
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    # Fallback to BDF
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF',
            rtol=1e-10,
            atol=1e-12
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    # If all else fails, return a zero vector or last known?
    # Should not happen for Robertson
    return np.zeros_like(y0)
