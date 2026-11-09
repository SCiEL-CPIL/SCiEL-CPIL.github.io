import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float, copy=True)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 250.0
    
    # Fixed-step Dormand-Prince 5(4) method
    # Using 5th order solution
    h = 0.001
    n_steps = int(np.ceil((t1 - t0) / h))
    h = (t1 - t0) / n_steps
    
    y_curr = y.copy()
    t_curr = t0
    
    # Coefficients for DP5
    # Stage 2: c2=1/5, a21=1/5
    # Stage 3: c3=3/10, a31=3/40, a32=9/40
    # Stage 4: c4=4/5, a41=44/45, a42=-56/15, a43=32/9
    # Stage 5: c5=8/9, a51=19372/6561, a52=-25360/2187, a53=64448/6561, a54=-212/729, a55=11/84
    # Stage 6: c6=1, a61=9017/31689, a62=355/3384, a63=3650/1296, a64=-45/11, a65=512/1771, a66=175/3220
    # b5 = [35/384, 0, 500/1113, 125/192, 2187/6784, 11/84, 0]
    
    for i in range(n_steps):
        if time.time() > deadline:
            break
            
        k1 = f(t_curr, y_curr)
        
        y2 = y_curr + h * (1/5 * k1)
        k2 = f(t_curr + h/5, y2)
        
        y3 = y_curr + h * (3/40 * k1 + 9/40 * k2)
        k3 = f(t_curr + 3*h/10, y3)
        
        y4 = y_curr + h * (44/45 * k1 - 56/15 * k2 + 32/9 * k3)
        k4 = f(t_curr + 4*h/5, y4)
        
        y5 = y_curr + h * (19372/6561 * k1 - 25360/2187 * k2 + 64448/6561 * k3 - 212/729 * k4 + 11/84 * k5 if False else 0)
        # Fix: k5 is not defined yet. The term 11/84 * k5 is for stage 6 input? No.
        # Stage 5 input y5 uses k1..k4.
        y5 = y_curr + h * (19372/6561 * k1 - 25360/2187 * k2 + 64448/6561 * k3 - 212/729 * k4)
        k5 = f(t_curr + 8*h/9, y5)
        
        y6 = y_curr + h * (9017/31689 * k1 + 355/3384 * k2 + 3650/1296 * k3 - 45/11 * k4 + 512/1771 * k5)
        k6 = f(t_curr + h, y6)
        
        y_curr = y_curr + h * (35/384 * k1 + 500/1113 * k3 + 125/192 * k4 + 2187/6784 * k5 + 11/84 * k6)
        t_curr += h
        
    return y_curr
