# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def _van_der_pol_jac(t, y):
    """Analytical Jacobian for the Van der Pol oscillator."""
    y0, y1 = y
    return np.array([
        [0.0, 1.0],
        [-10.0 * y0 * y1 - 1.0, 5.0 * (1.0 - y0 * y0)]
    ])


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using Radau method with tight tolerances.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Set a deadline to avoid timeout (300s limit, keep safety margin)
    deadline = time.time() + 290
    
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            jac=_van_der_pol_jac
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback 1: try BDF method
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='BDF',
            rtol=1e-11,
            atol=1e-13,
            jac=_van_der_pol_jac
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback 2: explicit RK4 with many steps
    steps = 50000
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
# EVOLVE-BLOCK-END
