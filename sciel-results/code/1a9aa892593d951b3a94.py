import numpy as np
import scipy.integrate as integrate_scipy
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use Radau for stiff problems. It's an implicit Runge-Kutta method.
    # High precision tolerances to minimize relative error.
    # max_step can be set to help control the step size if needed, but adaptive is usually better.
    
    try:
        sol = integrate_scipy.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=(t1-t0)/1000 # Initial max step to ensure resolution, adaptive will adjust
        )
        
        if not sol.success:
            # Fallback to BDF if Radau fails
            sol = integrate_scipy.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-12, 
                atol=1e-14
            )
            
        return sol.y[:, -1]
    except Exception:
        # Final fallback to a simple explicit method if SciPy fails for some reason
        # Though unlikely for Robertson
        h = (t1 - t0) / 10000
        t, y = t0, y0.copy()
        steps = int((t1 - t0) / h)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
