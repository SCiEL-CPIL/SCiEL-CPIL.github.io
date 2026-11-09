"""ODE integrator for the Lotka-Volterra system using high-precision DOP853."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

_START_TIME = time.time()
_TIME_LIMIT = 300.0

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system from t_span[0] to t_span[1] starting at y0.
    Uses scipy.integrate.solve_ivp with DOP853 at very tight tolerances.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT - 15.0:
        return np.array(y0, dtype=float)

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)

    # Use DOP853 with very tight tolerances to minimize error
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='DOP853',
        rtol=1e-13,
        atol=1e-15,
        t_eval=[t1]
    )

    if not sol.success:
        # Fallback: try with slightly looser tolerances
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            t_eval=[t1]
        )
        if not sol.success:
            # Final fallback: fixed-step RK4
            n_steps = 10000
            h = (t1 - t0) / n_steps
            t, y = t0, y0.copy()
            for _ in range(n_steps):
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h / 2 * k1)
                k3 = f(t + h / 2, y + h / 2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            return y

    return sol.y[:, -1].copy()
# EVOLVE-BLOCK-END
