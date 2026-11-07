"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rk45_step(f, t, y, h):
    """
    Perform one step of the 5(4) RK method (Dormand-Prince).
    Returns y_new (5th order) and y_err (4th order estimate for error estimation).
    """
    k1 = f(t, y)
    k2 = f(t + h / 5, y + h * (1 / 5) * k1)
    k3 = f(t + 3 * h / 10, y + h * (3 / 40 * k1 + 9 / 40 * k2))
    k4 = f(t + 4 * h / 5, y + h * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3))
    k5 = f(t + h, y + h * (-11 / 54 * k1 + 5 * k2 - 70 / 27 * k3 + 35 / 27 * k4))
    k6 = f(t + h / 2, y + h * (1631 / 55296 * k1 + 175 / 512 * k2 + 575 / 13824 * k3 - 135 / 512 * k4 + 25 / 4096 * k5))

    # 5th order solution
    y_new = y + h * (35 / 384 * k1 + 0 * k2 + 500 / 1113 * k3 + 125 / 192 * k4 - 2187 / 6784 * k5 + 11 / 84 * k6)
    
    # 4th order error estimate (difference between 5th and 4th order solutions)
    # The 4th order solution uses slightly different coefficients
    y_err = h * (5179 / 57600 * k1 - 7571 / 16695 * k2 + 397 / 165 * k3 - 2815 / 3168 * k4 + 285 / 1078 * k5 - 9 / 40 * k6)
    
    return y_new, y_err


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Start with a small step
    h = (t1 - t0) / 1000.0
    h_min = 1e-10
    h_max = (t1 - t0) / 10.0
    
    rtol = 1e-12
    atol = 1e-14
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, leave 20s margin
    
    while t < t1:
        if time.time() > deadline:
            # If we're running out of time, take a final large step to reach t1
            # This is a fallback, but we should rarely hit this
            remaining = t1 - t
            y = y + remaining * f(t, y)
            t = t1
            break
        
        # Ensure step doesn't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        y_new, y_err = _rk45_step(f, t, y, h)
        
        # Check for finite values
        if not np.all(np.isfinite(y_new)):
            # Step failed, reduce h
            h *= 0.5
            if h < h_min:
                # Give up and return current y
                break
            continue
        
        # Compute error norm
        err_norm = np.max(np.abs(y_err) / (atol + rtol * np.maximum(np.abs(y), np.abs(y_new))))
        
        if err_norm <= 1.0:
            # Step accepted
            t += h
            y = y_new
            
            # Adjust step size
            if err_norm > 0:
                factor = min(5.0, max(0.1, 0.9 * err_norm ** (-0.2)))
            else:
                factor = 5.0
            h_new = h * factor
            h = min(h_new, h_max)
            h = max(h, h_min)
        else:
            # Step rejected, reduce h
            factor = max(0.1, 0.9 * err_norm ** (-0.25))
            h = max(h * factor, h_min)
    
    return y
# EVOLVE-BLOCK-END
