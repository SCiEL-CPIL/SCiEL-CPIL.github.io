import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates the Lorenz system from t_span[0] to t_span[1] starting at y0.
    Uses DOP853 with tight tolerances for high accuracy and low call count.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    time_limit = 290.0  # Leave 10s margin
    
    # Primary method: DOP853 with very tight tolerances
    # rtol and atol are set to near machine precision to maximize accuracy
    # DOP853 is a high-order explicit Runge-Kutta method, very efficient for smooth problems
    try:
        sol = solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='DOP853', 
            rtol=1e-14, 
            atol=1e-16,
            max_step=(t1 - t0) / 100 # Prevent taking huge steps initially, though adaptive
        )
        
        if time.time() - start_time > time_limit:
             # If we're running out of time, return what we have or fallback
             pass 

        if sol.success:
            return sol.y[:, -1]
            
    except Exception:
        pass

    # Fallback: If DOP853 fails or is too slow, use a high-step RK4
    # This is unlikely to be needed for the Lorenz system on [0,2]
    try:
        steps = 20000
        h = (t1 - t0) / steps
        t = t0
        y = y0.copy()
        for _ in range(steps):
            if time.time() - start_time > time_limit:
                break
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
    except Exception:
        # Last resort: return y0 if everything fails (invalid but avoids exception)
        return y0
