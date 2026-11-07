# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t_start = time.time()
    deadline = t_start + 280.0  # 300s limit, keep 20s margin
    y0 = np.array(y0, dtype=float)
    t0_span, t1_span = t_span
    
    try:
        sol = solve_ivp(f, (t0_span, t1_span), y0, method='DOP853',
                        rtol=1e-11, atol=1e-13)
        if sol.success and sol.y.shape[1] > 0:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
        if sol.y.shape[1] > 0:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: fixed-step RK4 with moderate steps
    steps = 1000
    h = (t1_span - t0_span) / steps
    t, y = t0_span, y0.copy()
    for i in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
