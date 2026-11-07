"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Strategy: Use DOP853 with tighter tolerances to minimize relative error (rel).
    # The score is dominated by -log10(rel). Reducing rel by 100x gains 2 points.
    # Increasing calls by 10x loses only 0.5 points.
    # Therefore, we prioritize accuracy over call count.
    
    # Attempt 1: High accuracy DOP853
    try:
        sol = solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='DOP853', 
            rtol=1e-13, 
            atol=1e-15
        )
        
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Attempt 2: Fallback to RK45 with tight tolerances if DOP853 fails
    try:
        sol = solve_ivp(
            f, 
            (t0, t1), 
            y0, 
            method='RK45', 
            rtol=1e-12, 
            atol=1e-14
        )
        
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
    except Exception:
        pass

    # Fallback: Simple RK4 with small steps if all else fails
    # This is a last resort to avoid 0 score
    if time.time() > deadline - 10.0:
        h = (t1 - t0) / 1000
        t = t0
        y = y0.copy()
        for _ in range(1000):
            if time.time() > deadline - 2.0:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y

    # Standard RK4 fallback
    steps = 5000
    h = (t1 - t0) / steps
    t = t0
    y = y0.copy()
    for _ in range(steps):
        if time.time() > deadline - 5.0:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
