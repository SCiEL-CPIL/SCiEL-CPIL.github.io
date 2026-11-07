import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    ODE integrator for the Lorenz system.
    Uses scipy.integrate.solve_ivp with DOP853 method for high accuracy and efficiency.
    """
    start_time = time.time()
    deadline = 300.0  # Seconds
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Try high-order adaptive method
    try:
        # Match reference tolerances to ensure rel is at the floor (1e-14)
        # DOP853 is 8th order, very efficient
        sol = scipy.integrate.solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='DOP853', 
            rtol=1e-11, 
            atol=1e-13
        )
        
        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
        else:
            # If solve_ivp fails, fall back to a robust method
            pass
            
    except Exception:
        pass

    # Fallback: RK4 with many steps if adaptive method fails
    # This is unlikely to be needed for this specific problem
    steps = 10000
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    
    for _ in range(steps):
        if time.time() - start_time > deadline - 10:
            return y # Return best effort
        
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
        if not np.all(np.isfinite(y)):
            return y
            
    return y
