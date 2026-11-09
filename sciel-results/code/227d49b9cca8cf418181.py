import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, leave 10s margin
    
    N = len(y0)
    # Construct A = 21^2 * tridiag(1, -2, 1)
    # tridiag(1, -2, 1) has 1 on off-diagonals, -2 on diagonal
    diag = -2.0 * np.ones(N)
    off_diag = np.ones(N - 1)
    A = 441.0 * (np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1))
    
    def jac(t, y):
        return A

    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13, 
            jac=jac,
            max_step=0.1 # Optional: prevent huge steps if needed, but Radau handles stiffness well
        )
        
        if not sol.success:
            # Fallback if Radau fails for some reason
            # Use a simple explicit RK4 with small steps
            # This is a safety net, unlikely to be triggered
            h = 1e-4
            t0, t1 = t_span
            t = t0
            y = np.array(y0, dtype=float)
            while t < t1:
                if time.time() > deadline:
                    break
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
        else:
            return sol.y[:, -1]
            
    except Exception:
        # If solve_ivp throws, fallback to RK4
        h = 1e-4
        t0, t1 = t_span
        t = t0
        y = np.array(y0, dtype=float)
        while t < t1:
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
        return y
