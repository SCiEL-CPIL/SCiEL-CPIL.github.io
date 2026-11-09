import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use Radau method which is suitable for stiff problems
    # Tight tolerances to minimize rel error
    sol = scipy.integrate.solve_ivp(
        f, 
        (t0, t1), 
        y0, 
        method='Radau', 
        rtol=1e-12, 
        atol=1e-14,
        t_eval=[t1]
    )
    
    if not sol.success:
        # Fallback to a more robust but potentially slower method if Radau fails
        # Though Radau rarely fails for this problem
        sol = scipy.integrate.solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='LSODA', 
            rtol=1e-12, 
            atol=1e-14,
            t_eval=[t1]
        )
        
    if not sol.success:
        # Last resort: fixed step RK4
        # But we should avoid this if possible
        h = (t1 - t0) / 10000
        t, y = t0, y0.copy()
        steps = int((t1 - t0) / h)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y

    return sol.y[:, -1]
