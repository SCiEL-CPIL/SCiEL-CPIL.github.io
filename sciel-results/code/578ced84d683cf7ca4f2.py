"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cash_karp_step(f, t, y, h):
    """Cash-Karp RK45 pair. Returns (y5, error)."""
    k1 = f(t, y)
    k2 = f(t + h / 5, y + h * (1/5) * k1)
    k3 = f(t + 3*h/10, y + h * (3/40 * k1 + 9/40 * k2))
    k4 = f(t + 4*h/5, y + h * (44/45 * k1 - 56/15 * k2 + 32/9 * k3))
    k5 = f(t + 8*h/9, y + h * (19372/6561 * k1 - 25360/2187 * k2 + 64448/6561 * k3 - 212/729 * k4))
    k6 = f(t + h, y + h * (9017/3168 * k1 - 355/33 * k2 + 46732/5247 * k3 + 49/176 * k4 + 5103/18656 * k5))
    
    y4 = y + h * (37/378 * k1 + 25/216 * k3 + 25/27 * k4 + 1/27 * k5)
    y5 = y + h * (51/140 * k1 + 75/14 * k3 + 1/4 * k4 - 51/140 * k5 + 1/14 * k6)
    err = y5 - y4
    return y5, err


def _adaptive_rk45(f, t0, t1, y0, rtol, atol, max_steps, deadline):
    """Adaptive Cash-Karp RK45."""
    t = t0
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 50.0
    h_min = 1e-14
    h_max = (t1 - t0) * 0.5
    n = len(y)
    
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        if t >= t1 - 1e-15:
            break
        
        if t + h > t1:
            h = t1 - t
        if h < h_min:
            h = h_min
        
        y_new, err = _cash_karp_step(f, t, y, h)
        
        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            continue
        
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_scaled = np.sqrt(np.mean((err / scale) ** 2))
        
        if err_scaled <= 1.0:
            t += h
            y = y_new
            if err_scaled > 1e-16:
                factor = 0.9 * (1.0 / err_scaled) ** 0.2
            else:
                factor = 2.0
            h = max(h_min, min(h * factor, h_max, max(t1 - t, h_min)))
        else:
            factor = 0.9 * (1.0 / err_scaled) ** 0.25
            h = max(h_min, h * factor)
    
    return y


def _rk4_fixed(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    deadline = time.time() + 270
    
    # Adaptive RK45 with very tight tolerances
    y = _adaptive_rk45(f, t0, t1, y0, rtol=1e-13, atol=1e-15, max_steps=200000, deadline=deadline)
    
    if y is not None and np.all(np.isfinite(y)):
        return y
    
    # Fallback
    y = _rk4_fixed(f, t0, t1, y0, 10000)
    if y is not None:
        return y
    
    return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
