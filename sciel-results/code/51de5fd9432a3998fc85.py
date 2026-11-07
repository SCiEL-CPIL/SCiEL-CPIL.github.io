"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Deadline guard: 300s limit, leave 5s buffer
    start_time = time.time()
    deadline = start_time + 295.0
    
    # RKF45 coefficients
    a2 = 1/4
    a3 = 3/8
    b21 = 3/32
    b22 = 9/32
    b31 = 1932/2197
    b32 = -7200/2197
    b33 = 7296/2197
    c4 = 1
    b41 = 439/216
    b42 = -8
    b43 = 3680/513
    b44 = -845/4104
    c5 = 8/9
    b51 = -8/27
    b52 = 2
    b53 = -3544/2565
    b54 = 1859/4104
    b55 = -11/40
    c6 = 1
    b61 = 16/135
    b62 = 0
    b63 = 6656/12825
    b64 = 28561/56430
    b65 = -9/50
    b66 = 2/55
    
    # Error estimation weights
    e1 = 37/3780
    e2 = 0
    e3 = -250/621
    e4 = 1250/1989
    e5 = -2187/6784
    e6 = 11/84
    e7 = -5/870
    
    h = (t1 - t0) / 1000.0
    t = t0
    min_h = 1e-10
    max_h = (t1 - t0) / 10.0
    rtol = 1e-12
    atol = 1e-14
    
    while t < t1 - 1e-14:
        if time.time() > deadline:
            return y
            
        if t + h > t1:
            h = t1 - t
        
        k1 = f(t, y)
        k2 = f(t + a2 * h, y + h * (b21 * k1))
        k3 = f(t + a3 * h, y + h * (b31 * k1 + b32 * k2))
        k4 = f(t + c4 * h, y + h * (b41 * k1 + b42 * k2 + b43 * k3))
        k5 = f(t + c5 * h, y + h * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4))
        k6 = f(t + c6 * h, y + h * (b61 * k1 + b63 * k3 + b64 * k4 + b65 * k5 + b66 * k6))
        
        # 5th order solution
        y_new = y + h * (b61 * k1 + b63 * k3 + b64 * k4 + b65 * k5 + b66 * k6)
        
        # Error estimation
        err = h * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k6)
        
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.sum((err / scale) ** 2)) / np.sqrt(len(y))
        
        if err_norm <= 1.0:
            t += h
            y = y_new
            if err_norm > 1e-10:
                h = 0.9 * h * (err_norm ** -0.2)
            else:
                h = h * 5.0
            h = min(max(h, min_h), max_h)
        else:
            h = max(0.9 * h * (err_norm ** -0.25), min_h)
            
    return y
# EVOLVE-BLOCK-END
