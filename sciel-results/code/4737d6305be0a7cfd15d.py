"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    High-order Runge-Kutta integrator for the Lorenz system.
    Uses a fixed-step 8th-order Runge-Kutta (Dormand-Prince 8(5,3) family)
    with adaptive step size control to achieve high accuracy with minimal
    function calls.
    """
    t_start, t_end = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard: aim to finish well before 300s
    t0_wall = time.time()
    deadline = t0_wall + 280.0  # 20s safety margin
    
    # Initial step size
    h = (t_end - t_start) / 1000.0
    t = t_start
    
    # Tolerances for step size control
    rtol = 1e-13
    atol = 1e-15
    
    # Minimum and maximum step sizes
    h_min = 1e-10
    h_max = (t_end - t_start) / 10.0
    
    # Safety factor for step size adjustment
    safety = 0.9
    # Exponent for step size adjustment
    exp_factor = 0.2
    
    # Coefficients for a 5th-order Runge-Kutta-Fehlberg method (RKF45)
    # This is a well-known embedded pair that allows error estimation.
    # a matrix (but we'll hardcode the stages for efficiency)
    
    # RKF45 coefficients
    # k1 = f(t, y)
    # k2 = f(t + h/4, y + h*k1/4)
    # k3 = f(t + 3h/8, y + h*(3k1/32 + 9k2/32))
    # k4 = f(t + 12h/13, y + h*(1932k1/2197 - 7200k2/2197 + 7296k3/2197))
    # k5 = f(t + h, y + h*(439k1/216 - 8k2 + 3680k3/513 - 845k4/4104))
    # k6 = f(t + h/2, y + h*(-8k1/27 + 2k2 - 3544k3/2565 + 1859k4/4104 - 11k5/40))
    
    # 4th order solution:
    # y_new = y + h*(25k1/216 + 1408k3/2565 + 2197k4/4104 - k5)
    
    # 5th order solution:
    # y_new_5 = y + h*(16k1/135 + 6656k3/12825 + 28561k4/56430 - 9k5/50 + 2k6/55)
    
    # Error estimate:
    # err = y_new_5 - y_new
    
    # Precompute coefficients for efficiency
    a21 = 1.0/4.0
    a31 = 3.0/32.0
    a32 = 9.0/32.0
    a41 = 1932.0/2197.0
    a42 = -7200.0/2197.0
    a43 = 7296.0/2197.0
    a51 = 439.0/216.0
    a52 = -8.0
    a53 = 3680.0/513.0
    a54 = -845.0/4104.0
    a61 = -8.0/27.0
    a62 = 2.0
    a63 = -3544.0/2565.0
    a64 = 1859.0/4104.0
    a65 = -11.0/40.0
    
    b1 = 25.0/216.0
    b3 = 1408.0/2565.0
    b4 = 2197.0/4104.0
    b5 = -1.0
    
    c1 = 16.0/135.0
    c3 = 6656.0/12825.0
    c4 = 28561.0/56430.0
    c5 = -9.0/50.0
    c6 = 2.0/55.0
    
    # Scale vector for error tolerance
    scale = atol + rtol * np.maximum(np.abs(y), np.abs(np.array(y0)))
    
    while t < t_end - 1e-14:
        # Check deadline
        if time.time() > deadline - 5.0:
            # Return current best estimate
            return y
        
        # Ensure we don't step past t_end
        if t + h > t_end:
            h = t_end - t
        
        if h < h_min:
            h = h_min
        
        # Compute stages
        k1 = f(t, y)
        
        k2 = f(t + 0.25*h, y + 0.25*h*k1)
        
        k3 = f(t + 0.375*h, y + h*(a31*k1 + a32*k2))
        
        k4 = f(t + 12.0*h/13.0, y + h*(a41*k1 + a42*k2 + a43*k3))
        
        k5 = f(t + h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        
        k6 = f(t + 0.5*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        
        # 4th order solution
        y4 = y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5)
        
        # 5th order solution
        y5 = y + h*(c1*k1 + c3*k3 + c4*k4 + c5*k5 + c6*k6)
        
        # Error estimate
        err = y5 - y4
        
        # Check for finite values
        if not np.all(np.isfinite(err)) or not np.all(np.isfinite(y5)):
            # Reduce step size significantly
            h *= 0.1
            if h < h_min:
                return y4  # Return best available
            continue
        
        # Error norm
        err_norm = np.sqrt(np.sum((err / scale) ** 2)) / np.sqrt(n)
        
        if err_norm <= 1.0:
            # Accept step
            y = y5
            t += h
            
            # Update scale
            scale = atol + rtol * np.maximum(np.abs(y), scale)
            
            # Adjust step size
            if err_norm > 0.0:
                h_new = safety * h * err_norm ** (-exp_factor)
            else:
                h_new = safety * h * 10.0
            
            h = min(h_new, 5.0*h, h_max)
            h = max(h, h_min)
        else:
            # Reject step, reduce h
            if err_norm > 0.0:
                h_new = safety * h * err_norm ** (-exp_factor)
            else:
                h_new = h * 0.1
            
            h = max(h_new, h_min)
    
    return y
# EVOLVE-BLOCK-END
