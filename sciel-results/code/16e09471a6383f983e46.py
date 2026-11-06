"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate as integrate_scipy
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded


def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem with high accuracy.
    
    The Robertson problem is:
    dy0/dt = -1e7*y0 + 3e7*y1^2
    dy1/dt = 1e7*y0 - 4e7*y1^2 - 3e10*y1*y2
    dy2/dt = 3e10*y1*y2
    
    We use Radau IIA (implicit RK) with tight tolerances for high accuracy.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y0, y1, y2 = y
        return np.array([
            [-1e7, 6e7*y1, 0],
            [1e7, -8e7*y1 - 3e10*y2, -3e10*y1],
            [0, 3e10*y2, 3e10*y1]
        ])
    
    # Use Radau with very tight tolerances
    # rtol=1e-12, atol=1e-14 should give excellent accuracy
    # max_step controls step size; we want accurate results
    sol = solve_ivp(
        f, t_span, y0,
        method='Radau',
        jac=jac,
        rtol=1e-12,
        atol=1e-14,
        max_step=(t1 - t0) / 1000,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback to BDF with even tighter tolerances
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            jac=jac,
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 1000,
            dense_output=False
        )
    
    if not sol.success:
        # Last resort: LSODA
        sol = solve_ivp(
            f, t_span, y0,
            method='LSODA',
            jac=jac,
            rtol=1e-10,
            atol=1e-12,
            dense_output=False
        )
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
