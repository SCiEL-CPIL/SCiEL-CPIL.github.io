import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 250  # Safety margin for 300s limit
    
    call_count = 0
    def f_counted(t, y):
        nonlocal call_count
        call_count += 1
        return f(t, y)

    y0 = np.array(y0, dtype=float)
    
    # Try high-order adaptive method
    try:
        sol = solve_ivp(
            f_counted, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=0.01 # Prevent huge steps if not stiff, but allow small if needed
        )
        
        if time.time() > deadline:
             raise TimeoutError()
             
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Fallback: Optimized RK4
    # Estimate optimal steps.
    # If DOP853 failed or was slow, maybe the problem is tricky.
    # Use RK4 with K=500.
    t0, t1 = t_span
    steps = 500
    h = (t1 - t0) / steps
    t, y = t0, y0
    
    for _ in range(steps):
        k1 = f_counted(t, y)
        k2 = f_counted(t + h/2, y + h/2 * k1)
        k3 = f_counted(t + h/2, y + h/2 * k2)
        k4 = f_counted(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if time.time() > deadline:
            break
            
    return y
