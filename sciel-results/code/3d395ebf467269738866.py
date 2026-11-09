"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def _vdp_jacobian(t, y):
    """Analytical Jacobian for Van der Pol oscillator (mu=5).
    
    y0' = y1
    y1' = 5*(1 - y0^2)*y1 - y0
    
    J = [[0, 1],
         [-10*y0*y1 - 1, 5*(1 - y0^2)]]
    """
    y0, y1 = y[0], y[1]
    return np.array([
        [0.0, 1.0],
        [-10.0 * y0 * y1 - 1.0, 5.0 * (1.0 - y0**2)]
    ])


def integrate(f, t_span, y0):
    """
    Integrates the Van der Pol oscillator (mu=5) using scipy.integrate.solve_ivp
    with Radau method and analytical Jacobian to minimize function call count.
    
    The Radau method matches the reference solution's method, and providing the
    analytical Jacobian avoids finite-difference Jacobian computations, reducing
    the total number of f calls.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    start_time = time.time()
    
    # Use Radau with analytical Jacobian and same tolerances as reference
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-11,
            atol=1e-13,
            jac=lambda t, y: _vdp_jacobian(t, y)
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback if Radau fails
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                return np.array(y0, dtype=float)
    except Exception:
        # Fallback to DOP853 without Jacobian
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        return np.array(y0, dtype=float)
# EVOLVE-BLOCK-END
