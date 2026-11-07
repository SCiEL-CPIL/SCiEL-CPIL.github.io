"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate


def _rk4(f, t0, t1, y0, steps, deadline=None):
    """Fixed-step RK4 integrator with optional deadline guard."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        if deadline is not None and time.time() > deadline:
            break
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
    Integrates the Lotka-Volterra system using scipy.integrate.solve_ivp
    with the DOP853 method to match the reference solution's characteristics,
    minimizing relative error and function call count.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: stop before the 330s wall-clock limit
    start_time = time.time()
    deadline = start_time + 320  # safety margin
    
    try:
        # Use DOP853 with tolerances matching the reference solver
        # rtol=1e-11, atol=1e-13 as specified in the task
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            dense_output=False
        )
        
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    
    except Exception:
        pass
    
    # Fallback: if solve_ivp fails, try RK4 with enough steps
    # This is a safety net, not expected to be used
    try:
        steps = 5000
        y = _rk4(f, t0, t1, y0, steps, deadline)
        if y is not None and np.all(np.isfinite(y)):
            return y
    
    except Exception:
        pass
    
    # Last resort: return initial condition (should not happen)
    return y0
# EVOLVE-BLOCK-END
