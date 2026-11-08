"""Adaptive RKF45 integrator for the fixed Lorenz IVP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 250.0  # 300s limit, leave 50s margin
    
    # Adaptive RKF45 coefficients
    # 4th order: b4 = [1/4, 0, 1/3, 0, 0, 0]
    # 5th order: b5 = [1/5, 0, 9/40, 0, 9/40, 1/8]
    # Error estimation uses difference between 4th and 5th order
    
    # We use a standard RKF45 table
    # c = [0, 1/4, 3/8, 12/13, 1, 1/2]
    # a = [
    #   [0, 0, 0, 0, 0, 0],
    #   [1/4, 0, 0, 0, 0, 0],
    #   [3/32, 9/32, 0, 0, 0, 0],
    #   [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
    #   [439/216, -8, 3680/513, -845/4104, 0, 0],
    #   [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
    # ]
    # b4 = [25/216, 0, 1408/2565, 2197/4104, -1/5, 0]
    # b5 = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]
    
    def rkf45_step(t, y, h):
        k1 = f(t, y)
        k2 = f(t + h/4, y + h/4 * k1)
        k3 = f(t + 3*h/8, y + h * (3/32 * k1 + 9/32 * k2))
        k4 = f(t + 12*h/13, y + h * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3))
        k5 = f(t + h, y + h * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4))
        k6 = f(t + h/2, y + h * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5))
        
        y4 = y + h * (25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4 - 1/5 * k5)
        y5 = y + h * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        
        err = y5 - y4
        return y5, err, 6  # 6 function calls per step
    
    # Tolerances
    rtol = 1e-12
    atol = 1e-14
    
    # Initial step size
    h = (t1 - t0) / 1000
    t = t0
    
    while t < t1:
        if time.time() > deadline:
            break
        
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
        
        y_next, err, calls = rkf45_step(t, y, h)
        
        # Check for non-finite
        if not np.all(np.isfinite(y_next)):
            h *= 0.5
            continue
        
        # Error norm
        err_norm = np.linalg.norm(err)
        scale = rtol * np.linalg.norm(y_next) + atol
        
        if err_norm <= scale:
            # Accept step
            y = y_next
            t += h
            
            # Adjust step size
            if err_norm > 1e-16 * scale:
                h *= min(5.0, max(0.1, 0.9 * (scale / err_norm) ** 0.2))
            else:
                h *= 2.0
        else:
            # Reject step
            h *= max(0.1, 0.9 * (scale / err_norm) ** 0.25)
        
        if h < 1e-15:
            break
            
    return y
# EVOLVE-BLOCK-END
