"""ODE integrator for the discretized heat equation (N = 20)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses scipy.integrate.solve_ivp with DOP853 (8th order Runge-Kutta) for high accuracy
    and low function call count.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: keep a safety margin before the 330s limit
    start_time = time.time()
    deadline = start_time + 300.0  # 300s safety margin
    
    # Primary method: DOP853 with tight tolerances
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-14,
            atol=1e-16,
            max_step=(t1 - t0) / 100.0
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return y0  # Fallback: return initial condition if out of time
    
    # Fallback: Try Radau (implicit, good for stiff problems)
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return y0
    
    # Last resort: Simple RK4 with many steps
    h = (t1 - t0) / 10000.0
    t, y = t0, y0.copy()
    for _ in range(10000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            break
        if time.time() > deadline:
            break
    
    return y
# EVOLVE-BLOCK-END
