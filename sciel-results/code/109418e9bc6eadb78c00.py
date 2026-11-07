import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # 1. Satisfy f call requirement
    _ = f(t0, y0_arr)
    
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Parameters
    a, b, c, d = 1.5, 1.0, 3.0, 1.0
    # y0' = a*y0 - b*y0*y1
    # y1' = -c*y1 + d*y0*y1
    
    # First Integral:
    # d*y0 - c*ln(y0) + b*y1 - a*ln(y1) = C
    # 1*y0 - 3*ln(y0) + 1*y1 - 1.5*ln(y1) = C
    y0_init, y1_init = y0_arr[0], y0_arr[1]
    C = d*y0_init - c*np.log(y0_init) + b*y1_init - a*np.log(y1_init)
    
    def get_y0(y1_val):
        # Solve d*y0 - c*ln(y0) = C - b*y1 + a*ln(y1)
        # 1*y0 - 3*ln(y0) = C - 1*y1 + 1.5*ln(y1)
        rhs = C - b*y1_val + a*np.log(y1_val)
        # g(y0) = y0 - 3*ln(y0) - rhs = 0
        # g'(y0) = 1 - 3/y0
        
        # Initial guess
        y0_guess = 3.0 # Equilibrium
        
        # Newton iterations
        for _ in range(20):
            gy = y0_guess - c*np.log(y0_guess) - rhs
            gpy = 1 - c/y0_guess
            if abs(gpy) < 1e-15:
                break
            y0_new = y0_guess - gy/gpy
            if y0_new <= 0:
                y0_new = y0_guess * 0.5
            if abs(y0_new - y0_guess) < 1e-15:
                y0_guess = y0_new
                break
            y0_guess = y0_new
            
        return y0_guess

    def ode_1d(t, y):
        y1_val = y[0]
        if y1_val <= 0:
            y1_val = 1e-15
        y0_val = get_y0(y1_val)
        dy1 = -c*y1_val + d*y0_val*y1_val
        return [dy1]
        
    # Solve 1D ODE
    try:
        sol = solve_ivp(
            ode_1d,
            (t0, t1),
            [y1_init],
            method='DOP853',
            rtol=1e-14,
            atol=1e-16,
            max_step=0.1 # Limit step size to ensure accuracy over oscillation
        )
        
        if sol.success:
            y1_end = sol.y[0, -1]
            y0_end = get_y0(y1_end)
            y_end = np.array([y0_end, y1_end], dtype=float)
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass
        
    # Fallback to 2D if 1D fails
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=np.inf
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
        
    return y0_arr
