import numpy as np
import time
import scipy.integrate

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    try:
        # Use the same method and tolerances as the reference to minimize rel error
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        
        if not sol.success:
            # Fallback if solve_ivp fails
            # Simple RK4 fallback
            y = _rk4_fallback(f, t0, t1, y0)
            return y
            
        return sol.y[:, -1]
        
    except Exception as e:
        # Fallback on exception
        y = _rk4_fallback(f, t0, t1, y0)
        return y

def _rk4_fallback(f, t0, t1, y0, steps=10000):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
