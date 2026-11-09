"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 300.0
    
    # Try adaptive RK45 with tight tolerances
    # Use a custom RK45 implementation to avoid scipy overhead and control f calls
    
    def rk45_step(t, y, h):
        """Single RK45 step (Dormand-Prince). Returns y_new, error estimate."""
        k1 = f(t, y)
        k2 = f(t + h/5, y + h*(1/5)*k1)
        k3 = f(t + 3*h/10, y + h*(3/40*k1 + 9/40*k2))
        k4 = f(t + 4*h/5, y + h*(44/45*k1 - 56/15*k2 + 32/5*k3))
        k5 = f(t + h, y + h*(-11/54*k1 + 5/2*k2 - 70/27*k3 + 35/9*k4))
        k6 = f(t + 5*h/7, y + h*(1631/55296*k1 + 175/512*k2 + 575/13824*k3 + 44275/110592*k4 + 253/4096*k5))
        
        # 5th order solution
        y_new = y + h*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
        # Error estimate (difference between 5th and 4th order)
        err = h*(71/57600*k1 - 71/16695*k3 + 71/1920*k4 - 17253/339200*k5 + 22/525*k6 - 1/40*k5)
        return y_new, err
    
    # Adaptive RK45 integration
    rtol = 1e-12
    atol = 1e-14
    h = (t1 - t0) / 1000  # initial step
    h_min = (t1 - t0) / 100000
    h_max = (t1 - t0) / 10
    
    t = t0
    y_cur = y.copy()
    
    while t < t1 - 1e-15:
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
        
        y_new, err = rk45_step(t, y_cur, h)
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(y_new)):
            # Reduce step and retry
            h = h * 0.5
            if h < h_min:
                # Give up, return current best
                break
            continue
        
        # Compute error norm
        scale = atol + rtol * np.maximum(np.abs(y_cur), np.abs(y_new))
        err_norm = np.sqrt(np.mean((err / scale) ** 2))
        
        if err_norm <= 1.0:
            # Accept step
            t = t + h
            y_cur = y_new
            
            # Adapt step size
            if err_norm > 1e-8:
                factor = min(5.0, max(0.1, 0.9 * err_norm ** (-0.2)))
            else:
                factor = 5.0
            h = h * factor
            h = min(h, h_max)
            h = max(h, h_min)
        else:
            # Reject step
            factor = max(0.1, 0.9 * err_norm ** (-0.25))
            h = h * factor
            h = max(h, h_min)
        
        # Check deadline
        if time.time() > deadline - 10.0:
            break
    
    return y_cur
# EVOLVE-BLOCK-END
