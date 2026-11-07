"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rk45_bs(f, t0, t1, y0, rtol=1e-12, atol=1e-14, max_steps=10000):
    """
    Adaptive RK45 integrator using Bogacki-Shampine coefficients.
    """
    t, y = t0, np.array(y0, dtype=float)
    # Initial step estimate
    h = (t1 - t0) / 100.0
    h_min = (t1 - t0) / max_steps
    h_max = (t1 - t0) / 10.0
    
    # Bogacki-Shampine coefficients
    a2 = 1/4
    a3 = 3/8
    a4 = 12/13
    a5 = 1
    a6 = 1/2
    
    b21 = 1/4
    b31 = 3/32
    b32 = 9/32
    b41 = 1932/2197
    b42 = -7200/2197
    b43 = 7296/2197
    b51 = 439/216
    b52 = -8
    b53 = 3680/513
    b54 = -845/4104
    b61 = -8/27
    b62 = 2
    b63 = -3544/2565
    b64 = 1859/4104
    b65 = -11/40
    
    # 4th order coefficients (for error estimation and next step)
    c1 = 25/216
    c2 = 1408/2565
    c3 = 2197/4104
    c4 = -1/5
    c5 = 0
    c6 = 0
    
    # 5th order coefficients (for the actual step update)
    d1 = 16/135
    d2 = 0
    d3 = 6656/12825
    d4 = 28561/56430
    d5 = -9/50
    d6 = 2/55

    for _ in range(max_steps):
        if t >= t1 - 1e-15 * abs(t1 - t0):
            break
        
        k1 = f(t, y)
        k2 = f(t + a2*h, y + h*(b21*k1))
        k3 = f(t + a3*h, y + h*(b31*k1 + b32*k2))
        k4 = f(t + a4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
        k5 = f(t + a5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        k6 = f(t + a6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
        
        # 4th order solution (used for error estimate)
        y4 = y + h*(c1*k1 + c2*k3 + c3*k4 + c4*k5 + c6*k6)
        # 5th order solution (accepted step)
        y5 = y + h*(d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6)
        
        # Error estimation
        # err = max(|y5 - y4| / (atol + rtol * max(|y4|, |y5|)))
        scale = atol + rtol * np.maximum(np.abs(y4), np.abs(y5))
        err_vec = np.abs(y5 - y4) / scale
        err = np.max(err_vec)
        
        if err <= 1.0 or h <= h_min:
            # Accept step
            t += h
            y = y5
            if not np.all(np.isfinite(y)):
                return None
            
            # Estimate next step size
            if err > 0:
                factor = 0.9 * err**(-0.2)
            else:
                factor = 5.0
            factor = min(factor, 5.0)
            factor = max(factor, 0.1)
            h = h * factor
        else:
            # Reject step and reduce h
            factor = 0.9 * err**(-0.2)
            factor = min(factor, 0.9)
            factor = max(factor, 0.1)
            h = h * factor
        
        if h > h_max:
            h = h_max
        if h < h_min:
            h = h_min

    return y if np.all(np.isfinite(y)) else None


def _rk4(f, t0, t1, y0, steps):
    """Fixed-step RK4 as fallback."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
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
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Try adaptive RK45 first
    y = _rk45_bs(f, t0, t1, y0, rtol=1e-12, atol=1e-14, max_steps=10000)
    
    if y is not None and np.all(np.isfinite(y)):
        return y
    
    # Fallback to fixed-step RK4 if adaptive failed or took too long
    if time.time() < deadline:
        y = _rk4(f, t0, t1, y0, 2000)
        if y is not None and np.all(np.isfinite(y)):
            return y
            
    # Last resort: return initial state if everything failed (should not happen)
    return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
