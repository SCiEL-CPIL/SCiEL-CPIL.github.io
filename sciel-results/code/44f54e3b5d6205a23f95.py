# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 300s limit
    
    # Primary: DOP853 with very tight tolerances
    try:
        sol = solve_ivp(f, (t0, t1), y0, method='DOP853', rtol=1e-14, atol=1e-16)
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Check if we have enough time for fallback
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Very tight time: use RK4 with fewer steps
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
            if time.time() > deadline - 5:
                break
        return y
    
    # Secondary: Try DOP853 with slightly looser tolerances if primary failed
    try:
        sol = solve_ivp(f, (t0, t1), y0, method='DOP853', rtol=1e-13, atol=1e-15)
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: RK4 with many steps
    n_steps = 20000
    h = (t1 - t0) / n_steps
    t, y = t0, y0.copy()
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if time.time() > deadline - 5:
            break
    return y
# EVOLVE-BLOCK-END
