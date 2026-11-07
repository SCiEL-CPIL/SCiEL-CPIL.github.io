"""Rewritten ODE integrator for the fixed Lorenz IVP using adaptive RKF45."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rkf45_step(f, t, y, h):
    """Perform one RKF45 step, returning (y4, y5, err_est, nfev)."""
    k1 = f(t, y)
    k2 = f(t + h / 4.0, y + h * (k1 / 4.0))
    k3 = f(t + 3.0 * h / 8.0, y + h * (3.0 * k1 / 32.0 + 9.0 * k2 / 32.0))
    k4 = f(t + 12.0 * h / 13.0, y + h * (1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0))
    k5 = f(t + h, y + h * (439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0))
    k6 = f(t + h / 2.0, y + h * (-8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0))

    y4 = y + h * (25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - 1.0 * k5 / 5.0)
    y5 = y + h * (16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0)
    
    err_est = np.abs(y5 - y4)
    return y5, err_est, 6


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Initial step size
    h = (t1 - t0) / 100.0
    t = t0
    
    # Tolerances
    rtol = 1e-11
    atol = 1e-13
    
    nfev = 0
    deadline = time.time() + 280.0  # Safety margin before 330s limit
    
    while t < t1:
        if time.time() > deadline:
            break
            
        # Ensure we don't overshoot
        if t + h > t1:
            h = t1 - t
        
        # Try the step
        y_new, err, nfev_step = _rkf45_step(f, t, y, h)
        nfev += nfev_step
        
        # Check for non-finite
        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            if h < 1e-12:
                return y
            continue
        
        # Error norm
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.sqrt(np.sum((err / scale) ** 2)) / np.sqrt(n)
        
        # Accept/reject
        if err_norm <= 1.0:
            t += h
            y = y_new
            # Control step size
            if err_norm > 1e-12:
                factor = 0.9 * err_norm ** (-0.2)
            else:
                factor = 5.0
            h = h * min(factor, 5.0)
        else:
            # Reject step, reduce h
            factor = 0.9 * err_norm ** (-0.25)
            h = h * max(factor, 0.1)
            if h < 1e-12:
                break
        
        # Safety check
        if not np.all(np.isfinite(y)):
            return y
            
    return y
# EVOLVE-BLOCK-END
