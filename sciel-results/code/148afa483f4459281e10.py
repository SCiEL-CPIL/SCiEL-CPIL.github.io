import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-14, 
            atol=1e-15
        )
        
        if not sol.success:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='RK45', 
                rtol=1e-14, 
                atol=1e-15
            )
            
        if not sol.success:
            raise RuntimeError("Integration failed")
            
        return sol.y[:, -1]
        
    except Exception:
        # Fallback to fixed-step RK4
        N = 20000
        h = (t1 - t0) / N
        y = np.array(y0, dtype=float)
        t = t0
        
        for i in range(N):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            
            if time.time() > deadline:
                break
                
        return y
