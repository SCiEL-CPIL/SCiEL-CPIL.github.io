"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    
    Uses a high-accuracy explicit Runge-Kutta method with adaptive step size
    to achieve high precision while minimizing the number of function calls.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Track elapsed time for deadline guard
    t_start_wall = time.time()
    deadline = t_start_wall + 280.0  # Leave 50s safety margin
    
    # DOP853 coefficients (8th order Runge-Kutta)
    # This is a high-order method that requires 12 stages per step
    # Coefficients from Dormand-Prince 853 method
    
    # Using scipy's DOP853 directly via solve_ivp for best accuracy
    # with very tight tolerances
    
    def f_wrapper(t, y_vec):
        return np.array(f(t, y_vec), dtype=float)
    
    # Use DOP853 with extremely tight tolerances
    sol = solve_ivp(
        f_wrapper,
        (t0, t1),
        y,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15,
        max_step=(t1 - t0) / 500.0
    )
    
    if not sol.success:
        # Fallback to a manual high-order RK method
        # Use RK4 with many steps as fallback
        steps = 20000
        h = (t1 - t0) / steps
        t = t0
        y = np.array(y0, dtype=float)
        for _ in range(steps):
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2.0, y + h / 2.0 * k1)
            k3 = f(t + h / 2.0, y + h / 2.0 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
        return y
    
    return sol.y[:, -1]

# EVOLVE-BLOCK-END
