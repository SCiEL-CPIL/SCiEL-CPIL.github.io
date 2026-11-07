# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Fixed-step Dormand-Prince 5(4)
    # Coefficients for DP5(4)
    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    a = [
        [0, 0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ]
    b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    b_star = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    
    # Choose number of steps to balance accuracy and calls
    # For Lorenz on [0, 2], 1000 steps should be very accurate
    n_steps = 2000
    h = (t1 - t0) / n_steps
    t = t0
    y = y0.copy()
    
    for i in range(n_steps):
        if time.time() > deadline - 5.0:
            break
            
        k1 = f(t, y)
        k2 = f(t + c[1]*h, y + h*(a[1][0]*k1))
        k3 = f(t + c[2]*h, y + h*(a[2][0]*k1 + a[2][1]*k2))
        k4 = f(t + c[3]*h, y + h*(a[3][0]*k1 + a[3][1]*k2 + a[3][2]*k3))
        k5 = f(t + c[4]*h, y + h*(a[4][0]*k1 + a[4][1]*k2 + a[4][2]*k3 + a[4][3]*k4))
        k6 = f(t + c[5]*h, y + h*(a[5][0]*k1 + a[5][1]*k2 + a[5][2]*k3 + a[5][3]*k4 + a[5][4]*k5))
        
        # Use 5th order weights for the update
        y = y + h * (b[0]*k1 + b[2]*k3 + b[3]*k4 + b[4]*k5 + b[5]*k6)
        t = t + h
        
    return y

def _fallback(f, t0, t1, y0):
    """Simple RK4 fallback if solve_ivp fails"""
    n_steps = 50000
    h = (t1 - t0) / n_steps
    t = t0
    y = np.array(y0, dtype=float)
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
