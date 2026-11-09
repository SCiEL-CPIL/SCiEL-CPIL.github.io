"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use LSODA which adapts between stiff (BDF) and non-stiff (Adams) methods.
    # For this non-stiff problem, it will likely use high-order Adams methods (up to order 12),
    # potentially reducing the number of steps/calls compared to DOP853 (order 8).
    sol = solve_ivp(f, t_span, y0, method='LSODA', rtol=1e-11, atol=1e-13)
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback to a robust fixed-step RK4 if solve_ivp fails
        steps = 5000
        h = (t1 - t0) / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                return y
        return y
# EVOLVE-BLOCK-END
