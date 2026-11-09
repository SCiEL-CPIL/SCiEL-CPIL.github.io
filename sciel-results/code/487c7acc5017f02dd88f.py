"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the harmonic oscillator ODE using DOP853 for high accuracy and efficiency.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # DOP853 is an 8th-order explicit Runge-Kutta method.
    # Using tight tolerances ensures we reach the error floor (1e-14) with minimal calls.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-13, 
        atol=1e-15,
        max_step=1.0
    )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback to a high-step RK4 if the solver fails (unlikely for this problem)
        steps = 1000
        h = (t1 - t0) / steps
        t, y = t0, y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        return y
# EVOLVE-BLOCK-END
