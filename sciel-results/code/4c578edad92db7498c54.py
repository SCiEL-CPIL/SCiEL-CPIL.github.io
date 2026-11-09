import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using DOP853.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use the same method and tolerances as the reference to minimize relative error
    # while keeping the number of function calls low.
    sol = scipy.integrate.solve_ivp(
        f, 
        (t0, t1), 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13,
        max_step=0.1  # Optional: might help stability or step count, but DOP853 is adaptive. 
                      # Leaving it out usually lets it take larger steps where possible.
    )
    
    if not sol.success:
        # Fallback: try a simpler method if DOP853 fails (unlikely for this problem)
        sol = scipy.integrate.solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='RK45', 
            rtol=1e-10, 
            atol=1e-12
        )
        
    if not sol.success:
        # Last resort: fixed step RK4
        steps = 1000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            if time.time() > deadline:
                break
        return y

    return sol.y[:, -1]
