# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit
    
    # Primary: DOP853 with tolerances matching the reference exactly
    # This minimizes both error (matching ref) and calls (not over-integrating)
    try:
        sol = solve_ivp(f, (t0, t1), y0, method='DOP853', rtol=1e-11, atol=1e-13)
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Check deadline before fallback
    if time.time() > deadline:
        # Return best effort - use a simple RK4 with moderate steps
        n_steps = 1000
        h = (t1 - t0) / n_steps
        t, y = t0, y0.copy()
        for _ in range(n_steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
    
    # Fallback: RK4 with more steps
    n_steps = 5000
    h = (t1 - t0) / n_steps
    t, y = t0, y0.copy()
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
