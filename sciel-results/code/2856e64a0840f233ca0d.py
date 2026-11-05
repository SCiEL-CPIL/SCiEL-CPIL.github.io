"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

# Global deadline to ensure we finish within time limits
_start_time = time.time()
_DEADLINE = _start_time + 280.0  # Leave 20s margin for overhead

def _remaining_time():
    return max(0.0, _DEADLINE - time.time())

def _is_linear_time_invariant(f, t_span, y0):
    """
    Probes if f(t, y) is linear in y and independent of t.
    Returns A matrix if detected, else None.
    """
    t0, t1 = t_span
    n = len(y0)
    
    # Check time independence: f(t0, y0) should equal f(t1, y0)
    # We use a small vector to save calls if possible, but y0 is available.
    # Actually, let's just check f(t0, y0) vs f(t1, y0)
    f_t0 = f(t0, y0)
    f_t1 = f(t1, y0)
    if not np.allclose(f_t0, f_t1, rtol=1e-10, atol=1e-12):
        return None

    # Check linearity: f(t, y1 + y2) == f(t, y1) + f(t, y2)
    # and f(t, alpha * y) == alpha * f(t, y)
    # We can construct A directly if we assume linearity and verify with a few points.
    # A[i, j] is f(t, e_j)[i]
    
    A = np.zeros((n, n))
    t_mid = (t0 + t1) / 2.0
    
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t_mid, e_j)
        
    # Verify linearity with random vectors
    # f(t, y) should be A @ y
    # Test with y0
    y_pred = A @ y0
    y_actual = f(t_mid, y0)
    if not np.allclose(y_pred, y_actual, rtol=1e-10, atol=1e-12):
        return None
        
    # Test with another random vector
    rng = np.random.default_rng(42)
    y_test = rng.standard_normal(n)
    y_pred_test = A @ y_test
    y_actual_test = f(t_mid, y_test)
    if not np.allclose(y_pred_test, y_actual_test, rtol=1e-10, atol=1e-12):
        return None
        
    return A

def _solve_linear(f, t_span, y0, A):
    """Solves linear ODE y' = A y using matrix exponential."""
    t0, t1 = t_span
    dt = t1 - t0
    M = expm(A * dt)
    return M @ y0

def _rk45_adaptive(f, t_span, y0, rtol=1e-10, atol=1e-12):
    """
    Adaptive Runge-Kutta-Fehlberg 4(5) method.
    Falls back if linear probe fails.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    # Initial step size estimate
    h = (t1 - t0) / 10.0
    if h <= 0:
        return y
        
    # Coefficients for RK45 (Bogacki-Shampine or Fehlberg)
    # Using standard Fehlberg coefficients
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
    
    c1 = 25/216
    c2 = 1408/2565
    c3 = 2197/4104
    c4 = -1
    c5 = 1/5
    c6 = 0
    
    d1 = 16/135
    d2 = 0
    d3 = 6656/12825
    d4 = 28561/56430
    d5 = -9/50
    d6 = 2/55
    
    max_steps = 10000
    min_h = 1e-12
    
    for _ in range(max_steps):
        if _remaining_time() < 5.0:
            # Return current best guess (last accepted step)
            break
            
        k1 = f(t, y)
        k2 = f(t + a2*h, y + h*b21*k1)
        k3 = f(t + a3*h, y + h*(b31*k1 + b32*k2))
        k4 = f(t + a4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
        k5 = f(t + a5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        k6 = f(t + a6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
        
        y_new = y + h*(c1*k1 + c2*k3 + c3*k4 + c4*k5 + c5*k6)
        y_new_high = y + h*(d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6)
        
        err = np.linalg.norm(y_new_high - y_new)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        
        if np.all(err <= scale):
            # Accept step
            y = y_new
            t += h
            if t >= t1 - 1e-14:
                break
            # Adjust step size
            if err < 1e-10:
                h = min(2.0 * h, (t1 - t) / 2)
            else:
                h = 0.9 * h * (scale / err) ** 0.2
                h = min(h, 2.0 * h)
        else:
            # Reject step
            h = max(min_h, 0.9 * h * (scale / err) ** 0.2)
            if h <= min_h:
                # If we can't make progress, break
                break
                
    return y

def integrate(f, t_span, y0):
    y0 = np.array(y0, dtype=float)
    
    # Try linear solution first
    A = _is_linear_time_invariant(f, t_span, y0)
    if A is not None:
        return _solve_linear(f, t_span, y0, A)
    else:
        # Fallback to adaptive RK45
        return _rk45_adaptive(f, t_span, y0)
# EVOLVE-BLOCK-END
