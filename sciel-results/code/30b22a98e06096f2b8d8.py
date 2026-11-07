import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Fixed step size
    # We need high accuracy. Let's use h = 1e-4.
    # 50,000 steps.
    # 6 calls per step -> 300,000 calls.
    # This is well within 2,000,000 limit.
    
    # RK45 Coefficients (Fehlberg)
    # k1 = f(t, y)
    # k2 = f(t + h/4, y + h/4 * k1)
    # k3 = f(t + 3h/8, y + h * (3/32 * k1 + 9/32 * k2))
    # k4 = f(t + 12h/13, y + h * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3))
    # k5 = f(t + h, y + h * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4))
    # k6 = f(t + h/2, y + h * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5))
    
    # 5th order update:
    # y_new = y + h * (25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4 - 1/5 * k5)
    
    h = 1e-4
    n_steps = int(np.ceil((t1 - t0) / h))
    # Adjust h to exactly reach t1
    h = (t1 - t0) / n_steps
    
    for i in range(n_steps):
        t = t0 + i * h
        k1 = f(t, y)
        k2 = f(t + 0.25 * h, y + 0.25 * h * k1)
        k3 = f(t + 0.375 * h, y + h * (3/32 * k1 + 9/32 * k2))
        k4 = f(t + 12/13 * h, y + h * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3))
        k5 = f(t + h, y + h * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4))
        k6 = f(t + 0.5 * h, y + h * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5))
        
        # 5th order update
        y = y + h * (25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4 - 1/5 * k5)
        
    return y
