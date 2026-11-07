import numpy as np
import scipy.integrate

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use DOP853 with tight tolerances
    sol = scipy.integrate.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-14, 
        atol=1e-16,
        max_step=(t1-t0)/10  # Limit max step to ensure accuracy, maybe not needed but safe
    )
    
    if not sol.success:
        # Fallback to RK4 if solve_ivp fails
        h = (t1 - t0) / 10000
        t, y = t0, y0
        for _ in range(10000):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
        
    return sol.y[:, -1]
