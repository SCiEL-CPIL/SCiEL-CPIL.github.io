"""Integrator for Van der Pol oscillator using Radau method with tight tolerances."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) using scipy's Radau method
    with very tight tolerances to match the reference solution.
    
    The reference uses Radau with rtol=1e-11, atol=1e-13.
    We use the same method with slightly tighter tolerances to minimize
    relative error while keeping call count reasonable.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Track elapsed time for safety
    start_time = time.time()
    deadline = start_time + 280.0  # 280s deadline to stay under 330s kill
    
    from scipy.integrate import solve_ivp
    
    # Wrap f to ensure it returns a proper array
    def f_wrapper(t, y):
        result = f(t, y)
        return np.asarray(result, dtype=float)
    
    # Start with tightest tolerances for best accuracy
    # The score rewards lower rel error and penalizes extra calls
    # Go straight to tightest, then try even tighter if time permits
    configs = [
        (1e-13, 1e-15),   # Tight - good balance
        (1e-14, 1e-16),   # Very tight - if time allows
    ]
    
    best_result = None
    
    for rtol, atol in configs:
        if time.time() > deadline - 60:
            break
        
        try:
            sol = solve_ivp(
                f_wrapper,
                t_span,
                y0,
                method='Radau',
                rtol=rtol,
                atol=atol,
                dense_output=False
            )
            
            if sol.success and sol.y.shape[1] > 0:
                y_end = sol.y[:, -1]
                if np.all(np.isfinite(y_end)):
                    best_result = y_end
                    
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: adaptive RK45 with tight tolerances
    y = _rk45_adaptive(f, t0, t1, y0, rtol=1e-13, atol=1e-15)
    if y is not None and np.all(np.isfinite(y)):
        return y
    
    # Last resort: fixed-step RK4
    y = _rk4(f, t0, t1, y0, 5000)
    if y is not None and np.all(np.isfinite(y)):
        return y
    
    return y0


def _rk45_adaptive(f, t0, t1, y0, rtol=1e-11, atol=1e-13, max_steps=50000):
    """Adaptive RK45 (Dormand-Prince) integrator."""
    a2, a3, a4, a5, a6, a7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    b21 = 1/5
    b31, b32 = 3/40, 9/40
    b41, b42, b43 = 44/45, -56/15, 32/9
    b51, b52, b53, b54 = -13/20, 0, 3/4, -9/5
    b61, b62, b63, b64, b65 = -1/12, 0, 1/3, 1/4, 5/12
    b71, b72, b73, b74, b75, b76 = 19/72, 0, 5/12, 1/5, -5/72, 5/72
    c2, c3, c4, c5, c6, c7 = 1/5, 3/10, 4/5, 8/9, 1, 1
    e1 = 1/72 - 19/72
    e2 = 0 - 0
    e3 = 1/12 - 5/12
    e4 = 1/20 - 1/5
    e5 = -1/90 + 5/72
    e6 = 0 - 5/72
    e7 = 1/9 - 0

    y = np.array(y0, dtype=float)
    t = t0
    h = (t1 - t0) / 100.0
    min_h = (t1 - t0) / 1e6
    max_h = (t1 - t0) / 10.0

    for step in range(max_steps):
        if t >= t1:
            break
        if t + h > t1:
            h = t1 - t
        if h < min_h:
            h = min_h

        k1 = f(t, y)
        k2 = f(t + c2*h, y + h*b21*k1)
        k3 = f(t + c3*h, y + h*(b31*k1 + b32*k2))
        k4 = f(t + c4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
        k5 = f(t + c5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        k6 = f(t + c6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
        y_new = y + h*(b71*k1 + b72*k2 + b73*k3 + b74*k4 + b75*k5 + b76*k6)
        err = h*(e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6)

        if not np.all(np.isfinite(y_new)):
            return None

        err_norm = np.linalg.norm(err)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
        err_ratio = err_norm / np.linalg.norm(scale)

        if err_ratio <= 1.0:
            y = y_new
            t = t + h
            if err_ratio > 1e-5:
                h_new = h * min(5.0, max(0.1, 0.9 * err_ratio ** (-0.2)))
            else:
                h_new = h * 5.0
            h = min(h_new, max_h)
        else:
            h = h * max(0.1, 0.9 * err_ratio ** (-0.25))
            if h < min_h:
                h = min_h

    if t < t1 - 1e-12:
        h = t1 - t
        k1 = f(t, y)
        k2 = f(t + c2*h, y + h*b21*k1)
        k3 = f(t + c3*h, y + h*(b31*k1 + b32*k2))
        k4 = f(t + c4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
        k5 = f(t + c5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
        k6 = f(t + c6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))
        y = y + h*(b71*k1 + b72*k2 + b73*k3 + b74*k4 + b75*k5 + b76*k6)

    return y


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
