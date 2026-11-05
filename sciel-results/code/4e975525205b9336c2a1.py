"""ODE solver for 100-point convection-diffusion IVP."""
import numpy as np
import time

def _dopri5_step(f, t, y, h):
    """One DOPRI5 (Dormand-Prince 5(4)) step. Returns y_new, err, nfev."""
    a2 = 1/5
    a3 = 3/10
    a4 = 4/5
    a5 = 8/9
    a6 = 1.0
    
    b1 = 0
    b2 = 5/72
    b3 = 0
    b4 = 10/27
    b5 = 5/27
    b6 = -5/72
    
    c2 = 1/5
    c3 = 3/10
    c4 = 4/5
    c5 = 8/9
    c6 = 1.0
    
    k1 = f(t, y)
    k2 = f(t + c2*h, y + h*(b1*k1))
    k3 = f(t + c3*h, y + h*(b2*k1 + b3*k2))
    k4 = f(t + c4*h, y + h*(b4*k1 + b5*k2 + b6*k3))
    k5 = f(t + c5*h, y + h*(0*k1 + 10/27*k2 + 25/27*k3 + -200/81*k4))
    k6 = f(t + c6*h, y + h*(1/36*k1 + 1/9*k2 + 5/12*k3 + -5/4*k4 + 71/108*k5))
    
    y_new = y + h*(5/72*k1 + 15/16*k3 + 5/12*k4 + -5/4*k5 + 71/60*k6)
    y_4th = y + h*(b1*k1 + b2*k2 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
    err = np.linalg.norm(y_new - y_4th, ord=np.inf)
    
    return y_new, err, 6


def _dopri5_adaptive(f, t0, t1, y0, rtol=1e-10, atol=1e-12, max_steps=50000, deadline=None):
    """Adaptive DOPRI5 with step size control."""
    t = t0
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 100  # initial guess
    total_fev = 0
    steps = 0
    
    while t < t1 - 1e-15:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        if steps >= max_steps:
            break
        
        # Don't overshoot
        if t + h > t1:
            h = t1 - t
        if h < 1e-15:
            break
        
        y_new, err, nfev = _dopri5_step(f, t, y, h)
        total_fev += nfev
        steps += 1
        
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_scaled = np.max(np.abs(y_new - y) / scale) if scale.size > 0 else err
        
        if not np.all(np.isfinite(y_new)) or np.max(np.abs(y_new)) > 1e12:
            # Reject step, reduce h
            h *= 0.5
            if h < 1e-16:
                break
            continue
        
        if err_scaled <= 1.0:
            # Accept step
            t = t + h
            y = y_new
            # Adjust step size
            if err_scaled < 1e-8:
                h_new = h * 5.0
            else:
                h_new = h * min(5.0, 0.9 * err_scaled ** (-0.2))
            h = min(h_new, t1 - t)
        else:
            # Reject step
            h = h * max(0.1, 0.9 * err_scaled ** (-0.25))
    
    return y, total_fev


def _rk45_fixed(f, t0, t1, y0, steps):
    """Fixed-step RK4 (4th order) as fallback."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2*k1)
        k3 = f(t + h/2, y + h/2*k2)
        k4 = f(t + h, y + h*k3)
        y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    deadline = time.time() + 280.0  # safety margin
    
    # Try adaptive DOPRI5 first
    y, fev = _dopri5_adaptive(f, t0, t1, y0, rtol=1e-10, atol=1e-12, 
                               max_steps=50000, deadline=deadline)
    
    if y is not None and np.all(np.isfinite(y)) and np.max(np.abs(y)) < 1e12:
        return y
    
    # Fallback: fixed-step RK4 with more steps
    y = _rk45_fixed(f, t0, t1, y0, 5000)
    if y is not None:
        return y
    
    # Last resort
    y = _rk45_fixed(f, t0, t1, y0, 2000)
    return y
