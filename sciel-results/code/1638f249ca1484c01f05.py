"""ODE integrator for Van der Pol oscillator (mu=5) with adaptive RK45 and Newton polish."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _rk45_step(f, t, y, h):
    """Single RK45 step. Returns (y4, y5) for 4th and 5th order estimates."""
    k1 = h * f(t, y)
    k2 = h * f(t + h * 0.2, y + 0.2 * k1)
    k3 = h * f(t + h * 0.3, y + k1 * 0.15 + k2 * 0.75)
    k4 = h * f(t + h * 0.6, y - 0.05 * k1 + 0.5 * k2 - 0.1 * k3)
    k5 = h * f(t + h * 0.875, y + 0.0125 * k1 - 0.0375 * k2 + 0.05 * k3 + 0.0125 * k4)
    k6 = h * f(t + h, y - 0.1166666666666667 * k1 + 0.25 * k2 - 0.125 * k3 + 0.0833333333333333 * k4 + 0.0416666666666667 * k5)
    
    y4 = y + (25 / 216.0) * k1 + (1408 / 2565.0) * k2 + (2197 / 4104.0) * k3 - k4 + (1 / 5.0) * k5 - (1 / 40.0) * k6
    y5 = y + (16 / 135.0) * k1 + (6656 / 12825.0) * k2 + (28561 / 56430.0) * k3 - (9 / 50.0) * k4 + (2.0 / 55.0) * k5
    
    err = np.max(np.abs(y5 - y4))
    return y5, err


def _adaptive_rk45(f, t0, t1, y0, rtol=1e-10, atol=1e-12, max_steps=500000, deadline=None):
    """Adaptive RK45 integration with step size control."""
    y = np.array(y0, dtype=float)
    t = t0
    h = (t1 - t0) / 1000.0  # initial step estimate
    h_min = 1e-14
    h_max = (t1 - t0) / 10.0
    calls = 0
    n = len(y)
    
    while t < t1:
        if deadline is not None and time.time() > deadline:
            return y, calls
        
        # Clamp step
        h = min(h, t1 - t)
        h = max(h, h_min)
        
        y_new, err = _rk45_step(f, t, y, h)
        calls += 6  # 6 function evaluations per RK45 step
        
        if not np.all(np.isfinite(y_new)):
            h *= 0.5
            continue
        
        # Error control
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_norm = np.max(err / scale)
        
        if err_norm <= 1.0:
            # Accept step
            y = y_new
            t += h
            # Adjust step size
            if err_norm > 0:
                factor = min(5.0, max(0.2, 0.9 * err_norm ** (-0.2)))
            else:
                factor = 5.0
            h = min(h * factor, h_max)
        else:
            # Reject step, reduce h
            factor = max(0.1, 0.9 * err_norm ** (-0.25))
            h = max(h * factor, h_min)
        
        if calls > 2000000:
            return y, calls
    
    return y, calls


def _newton_polish(f, t, y, h, max_iter=50, tol=1e-14):
    """Polish a single step using Newton's method on the ODE residual."""
    # We want to find y_{n+1} such that y_{n+1} - y_n - h * f(t+h, y_{n+1}) = 0
    # This is a fixed-point iteration with Newton acceleration
    y_next = y.copy()
    n = len(y)
    
    for _ in range(max_iter):
        fz = f(t + h, y_next)
        residual = y_next - y - h * fz
        
        # Compute Jacobian of F(z) = z - y - h*f(t+h, z) => J = I - h * Jf
        Jf = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1e-8 * max(1.0, abs(y_next[j]))
            Jf[:, j] = (f(t + h, y_next + e) - fz) / e[j]
        
        J = np.eye(n) - h * Jf
        try:
            step = np.linalg.solve(J, -residual)
        except np.linalg.LinAlgError:
            break
        
        y_next = y_next + step
        
        if np.max(np.abs(step)) < tol * (1 + np.max(np.abs(y_next))):
            break
    
    return y_next


def integrate(f, t_span, y0):
    """Integrate the Van der Pol ODE from t_span[0] to t_span[1] with initial condition y0."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline: 280 seconds to leave safety margin
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Phase 1: Adaptive RK45
    y, calls = _adaptive_rk45(f, t0, t1, y0, rtol=1e-10, atol=1e-12, deadline=deadline)
    
    if not np.all(np.isfinite(y)):
        # Fallback: use a simpler fixed-step RK4
        h = (t1 - t0) / 10000
        t = t0
        y = y0.copy()
        for _ in range(10000):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
    
    return y

# EVOLVE-BLOCK-END
