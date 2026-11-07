"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _rk45(f, t0, t1, y0, rtol=1e-12, atol=1e-14, max_steps=50000):
    """Adaptive RK45 (Bogacki-Shampine) integrator."""
    t, y = t0, np.array(y0, dtype=float)
    h = (t1 - t0) / 100.0
    h_min = (t1 - t0) / max_steps
    h_max = (t1 - t0) / 10.0
    
    # RK45 coefficients (Bogacki-Shampine)
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
    
    # 4th order coefficients
    c1 = 25/216
    c2 = 1408/2565
    c3 = 2197/4104
    c4 = -1/5
    c5 = 0
    c6 = 0
    
    # 5th order coefficients
    d1 = 16/135
    d2 = 0
    d3 = 6656/12825
    d4 = 28561/56430
    d5 = -9/50
    d6 = 2/55
    
    for _ in range(max_steps):
        if t >= t1 - 1e-15:
            break
        
        k1 = f(t, y)
        k2 = f(t + a2*h, y + h*(b21*k1))
        k3 = f(t + a3*h, y + h*(b31*k1 + b32*k2))
        k4 = f(t + a4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
        k5 = f(t + a5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        k6 = f(t + a6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
        
        y4 = y + h*(c1*k1 + c2*k3 + c3*k4 + c4*k5 + c6*k6)
        y5 = y + h*(d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6)
        
        err = np.max(np.abs(y5 - y4) / (atol + rtol * np.maximum(np.abs(y4), np.abs(y5))))
        
        if err <= 1.0 or h <= h_min:
            t += h
            y = y5
            if not np.all(np.isfinite(y)):
                return None
        else:
            # Reduce step size
            factor = 0.9 * err**(-0.2)
            factor = min(factor, 5.0)
            factor = max(factor, 0.1)
            h = h * factor
        
        if h > h_max:
            h = h_max
        if h < h_min:
            h = h_min
    
    return y if np.all(np.isfinite(y)) else None


def _backward_euler(f, t0, t1, y0, steps):
    # Implicit Euler with a finite-difference Jacobian and a few Newton iterations (handles stiff problems).
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    for _ in range(steps):
        z = y.copy()
        for _newton in range(4):
            fz = f(t + h, z)
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-7 * max(1.0, abs(z[j]))
                J[:, j] = (f(t + h, z + e) - fz) / e[j]
            step = np.linalg.solve(np.eye(n) - h * J, -(z - y - h * fz))
            z = z + step
            if np.linalg.norm(step) <= 1e-10 * (1 + np.linalg.norm(z)):
                break
        y, t = z, t + h
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Try adaptive RK45 first
    y = _rk45(f, t0, t1, y0, rtol=1e-12, atol=1e-14, max_steps=50000)
    
    if y is not None and np.all(np.isfinite(y)):
        return y
    
    # Fall back to fixed-step RK4 if adaptive failed
    y = _rk4(f, t0, t1, y0, 2000)
    if y is not None and np.all(np.isfinite(y)):
        return y
    
    # Final fallback: implicit Euler
    y = _backward_euler(f, t0, t1, y0, 2000)
    return y if np.all(np.isfinite(y)) else np.array(y0, dtype=float)

def _rk4(f, t0, t1, y0, steps):
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
# EVOLVE-BLOCK-END
