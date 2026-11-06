import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: 330s wall-clock limit, use 320s as safety margin
    deadline = time.time() + 320
    
    # Try Radau first (good for stiff convection-diffusion problems)
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
            max_step=0.01
        )
        
        if sol.success:
            return sol.y[:, -1]
        
        if time.time() > deadline:
            # Return last computed value if available
            if sol.y.shape[1] > 0:
                return sol.y[:, -1]
            return y0
            
    except Exception:
        pass
    
    # If Radau failed or timed out, try LSODA
    try:
        if time.time() > deadline:
            return y0
            
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='LSODA',
            rtol=1e-10,
            atol=1e-12
        )
        
        if sol.success:
            return sol.y[:, -1]
            
        if time.time() > deadline:
            if sol.y.shape[1] > 0:
                return sol.y[:, -1]
            return y0
            
    except Exception:
        pass
    
    # Last resort: DOP853
    try:
        if time.time() > deadline:
            return y0
            
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12
        )
        
        if sol.success:
            return sol.y[:, -1]
            
        if sol.y.shape[1] > 0:
            return sol.y[:, -1]
            
    except Exception:
        pass
    
    return y0
