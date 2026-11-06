"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Use DOP853 (8th order) with tight tolerances for high accuracy
    # This method is very efficient for smooth problems like convection-diffusion
    sol = scipy.integrate.solve_ivp(
        f,
        (t0, t1),
        y0_arr,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14,
        max_step=np.inf
    )
    
    # If the solver fails or returns no solution, fall back to a simple RK4
    if not sol.success or sol.y.shape[1] == 0:
        # Fallback: simple RK4 with many steps
        n_steps = 10000
        h = (t1 - t0) / n_steps
        t = t0
        y = y0_arr.copy()
        for _ in range(n_steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
