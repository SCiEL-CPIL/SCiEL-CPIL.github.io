"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    # Use the same method and tolerances as the reference to minimize relative error
    # and leverage adaptive step size control to minimize function calls.
    sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-13, atol=1e-15)
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback to a robust fixed-step RK4 if solve_ivp fails
        # This is unlikely to be triggered for this smooth problem.
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
