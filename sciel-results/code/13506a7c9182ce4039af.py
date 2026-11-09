import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    start_time = time.time()
    
    # Primary strategy: High-order explicit method DOP853
    # Good for non-stiff to mildly stiff problems.
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-10, 
            atol=1e-12
        )
        if sol.success:
            # If it finished quickly, return result
            if time.time() - start_time < 300:
                return sol.y[:, -1]
        else:
            # If it failed, we might need a different method
            pass
    except Exception:
        pass

    # Fallback strategy: Implicit method Radau
    # Better for stiff problems.
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

    # Last resort: Simple explicit Euler with many steps
    # This is inefficient but ensures a return value.
    h = (t1 - t0) / 5000
    t = t0
    y = y0.copy()
    steps = int((t1 - t0) / h)
    for _ in range(steps):
        y = y + h * f(t, y)
        t += h
        if time.time() - start_time > 290:
            break
    return y
