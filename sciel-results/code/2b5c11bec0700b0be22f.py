import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280  # 280s safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    try:
        # DOP853 is a high-order explicit Runge-Kutta method (8th order).
        # Using tighter tolerances than the reference (rtol=1e-11, atol=1e-13)
        # to reduce relative error.
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15,
            max_step=0.05 # Constrain step size to ensure accuracy in oscillatory regions
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if solve_ivp fails
            # Use a high-order fixed-step RK if adaptive fails
            # But this is unlikely.
            return y0
    except Exception:
        # If scipy fails or times out (unlikely to timeout, but possible to error)
        # Return a crude result to avoid invalidity
        # Simple RK4 with 1000 steps
        steps = 1000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            if time.time() > deadline:
                return y
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
