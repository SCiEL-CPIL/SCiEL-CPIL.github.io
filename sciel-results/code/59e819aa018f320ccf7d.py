"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    deadline = time.time() + 320  # safety margin before 330s kill
    y0 = np.array(y0, dtype=float)
    
    # Try adaptive high-order DOP853 with tight tolerances
    # This is a structural change: adaptive step-size vs fixed-step
    try:
        sol = solve_ivp(
            f, (t0, t1), y0,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 100,
            dense_output=False
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: fixed-step RK4 with many steps
    if time.time() < deadline:
        steps = 5000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
                break
        return y
    
    return y0
# EVOLVE-BLOCK-END
