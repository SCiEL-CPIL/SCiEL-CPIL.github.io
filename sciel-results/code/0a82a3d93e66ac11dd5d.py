"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system y0' = 1.5*y0 - y0*y1, y1' = -3*y1 + y0*y1
    from t_span[0] to t_span[1] with initial condition y0.
    Returns the state at t_span[1].
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 300s limit, keep 10s margin
    
    # Use a high-order Runge-Kutta method with adaptive step size.
    # We'll implement a DOP853-like method, but for simplicity and speed,
    # we'll use a fixed-step high-order RK with many steps, or an adaptive RK45.
    
    # Let's use an adaptive RK45 (Bogacki-Shampine) with tight tolerances.
    rtol = 1e-13
    atol = 1e-15
    
    # Bogacki-Shampine coefficients
    a2 = 1/4
    a3 = 3/8
    a4 = 12/13
    a5 = 1.0
    a6 = 1/2
    
    b21 = 1/4
    b31 = 3/32
    b32 = 9/32
    b41 = 1932/2197
    b42 = -7200/2197
    b43 = 7296/2197
    b51 = 439/216
    b52 = -8.0
    b53 = 3680/513
    b54 = -845/4104
    b61 = -8/27
    b62 = 2.0
    b63 = -3544/2565
    b64 = 1859/4104
    b65 = -11/40
    
    c1 = 25/216
    c2 = 1408/2565
    c3 = 2197/4104
    c4 = -1/5
    
    d1 = 16/135
    d2 = 0.0
    d3 = 6656/12825
    d4 = 28561/56430
    d5 = -9/50
    d6 = 2/55
    
    # Initial step size estimate
    t = t0
    h = (t1 - t0) / 1000.0
    h_min = 1e-12
    h_max = (t1 - t0) / 10.0
    
    # Safety factor
    safety = 0.9
    min_factor = 0.1
    max_factor = 10.0
    
    while t < t1:
        # Ensure we don't overshoot
        if t + h > t1:
            h = t1 - t
        
        # Check deadline
        if time.time() > deadline:
            break
        
        # RK45 step
        k1 = f(t, y)
        k2 = f(t + a2 * h, y + h * (b21 * k1))
        k3 = f(t + a3 * h, y + h * (b31 * k1 + b32 * k2))
        k4 = f(t + a4 * h, y + h * (b41 * k1 + b42 * k2 + b43 * k3))
        k5 = f(t + a5 * h, y + h * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4))
        k6 = f(t + a6 * h, y + h * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5))
        
        # 4th order solution
        y4 = y + h * (c1 * k1 + c2 * k3 + c3 * k4 + c4 * k5)
        
        # 5th order solution
        y5 = y + h * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6)
        
        # Error estimate
        err = np.linalg.norm(y5 - y4)
        scale = atol + rtol * np.maximum(np.abs(y4), np.abs(y5))
        err_scaled = np.max(err / scale) if len(err) > 0 else 0.0
        
        if err_scaled <= 1.0:
            # Accept step
            y = y5
            t = t + h
            
            # Adjust step size
            if err_scaled < 1e-10:
                factor = max_factor
            else:
                factor = safety * err_scaled ** (-0.2)
            factor = min(max_factor, max(min_factor, factor))
            h = h * factor
            h = min(h, h_max)
            h = max(h, h_min)
        else:
            # Reject step
            factor = safety * err_scaled ** (-0.25)
            factor = max(min_factor, factor)
            h = h * factor
            h = max(h, h_min)
    
    return y
# EVOLVE-BLOCK-END
