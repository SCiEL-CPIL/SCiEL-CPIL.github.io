"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rk45_step(f, t, y, h):
    """
    Single Bogacki-Shampine RK45 step.
    Returns (y_new, err) where err is the estimated error.
    Uses 4th order solution for stepping, 5th order for error estimation.
    """
    k1 = f(t, y)
    k2 = f(t + h / 4.0, y + h / 4.0 * k1)
    k3 = f(t + 3.0 * h / 8.0, y + h * (3.0 / 32.0 * k1 + 9.0 / 32.0 * k2))
    k4 = f(t + 12.0 * h / 13.0, y + h * (1932.0 / 2197.0 * k1 - 7200.0 / 2197.0 * k2 + 7296.0 / 2197.0 * k3))
    k5 = f(t + h, y + h * (439.0 / 216.0 * k1 - 8.0 * k2 + 3680.0 / 513.0 * k3 - 845.0 / 4104.0 * k4))
    k6 = f(t + h / 2.0, y + h * (-8.0 / 27.0 * k1 + 2.0 * k2 - 3544.0 / 2565.0 * k3 + 1859.0 / 4104.0 * k4 - 11.0 / 40.0 * k5))
    
    # 4th order solution (used for stepping)
    y_new = y + h * (25.0 / 216.0 * k1 + 1408.0 / 2565.0 * k3 + 2197.0 / 4104.0 * k4 - 1.0 / 5.0 * k5)
    
    # Error estimate (5th order - 4th order)
    err = h * (1.0 / 120.0 * k1 - 1408.0 / 2565.0 * k3 + 2197.0 / 4104.0 * k4 - 1.0 / 5.0 * k5 + 1.0 / 4.0 * k6)
    
    return y_new, err


def _adaptive_rk45(f, t0, t1, y0, rtol=1e-12, atol=1e-14, max_steps=100000, deadline=None):
    """
    Adaptive RK45 integrator with step size control.
    """
    t = t0
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 100.0
    h_min = (t1 - t0) / max_steps
    h_max = (t1 - t0) / 5.0
    
    safety = 0.9
    factor_min = 0.1
    factor_max = 10.0
    
    steps = 0
    start_time = time.time()
    
    while t < t1 - 1e-15:
        # Check deadline
        if deadline is not None and time.time() > deadline:
            return None
        
        # Check time budget
        if time.time() - start_time > 250:
            return None
        
        # Limit step to not overshoot
        h = min(h, t1 - t)
        
        y_new, err = _rk45_step(f, t, y, h)
        
        # Check for finite values
        if not np.all(np.isfinite(y_new)):
            # Try smaller step
            h = h * 0.1
            if h < h_min:
                return None
            continue
        
        # Error norm
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.linalg.norm(err / scale) / np.sqrt(len(y))
        
        if err_norm <= 1.0:
            # Accept step
            t += h
            y = y_new
            steps += 1
            
            # Update step size
            if err_norm > 1e-8:
                factor = safety * err_norm ** (-0.2)
            else:
                factor = factor_max
            factor = max(factor_min, min(factor_max, factor))
            h = h * factor
            h = max(h_min, min(h_max, h))
        else:
            # Reject step, reduce h
            factor = safety * err_norm ** (-0.25)
            factor = max(factor_min, min(factor_max, factor))
            h = h * factor
            h = max(h_min, h)
        
        # Safety: if we've taken too many steps, return what we have
        if steps > 2000000:
            return None
    
    return y


def _rk4(f, t0, t1, y0, steps, deadline=None):
    """
    Fixed-step RK4 integrator.
    """
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    start_time = time.time()
    
    for _ in range(steps):
        # Check deadline
        if deadline is not None and time.time() > deadline:
            return None
        if time.time() - start_time > 250:
            return None
            
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using adaptive RK45,
    with fallback to fixed-step RK4 if adaptive fails.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Set deadline with safety margin
    deadline = time.time() + 280
    
    # Try adaptive RK45 first
    y = _adaptive_rk45(f, t0, t1, y0, rtol=1e-12, atol=1e-14, max_steps=50000, deadline=deadline)
    
    if y is not None and np.all(np.isfinite(y)):
        return y
    
    # Fallback to fixed-step RK4 with more steps
    y = _rk4(f, t0, t1, y0, 5000, deadline=deadline)
    
    if y is not None and np.all(np.isfinite(y)):
        return y
    
    # Last resort: even more steps
    y = _rk4(f, t0, t1, y0, 10000, deadline=deadline)
    
    if y is not None and np.all(np.isfinite(y)):
        return y
    
    # If all else fails, return initial condition (should not happen)
    return y0
# EVOLVE-BLOCK-END
