# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    def f_wrapped(t, y):
        return f(t, y)
    
    sol = solve_ivp(f_wrapped, (t0, t1), y0, method='DOP853', rtol=1e-12, atol=1e-14)
    if sol.success:
        return sol.y[:, -1]
    
    # Fallback: simple RK4 with many steps
    h = (t1 - t0) / 20000
    t, y = t0, y0.copy()
    for _ in range(20000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
