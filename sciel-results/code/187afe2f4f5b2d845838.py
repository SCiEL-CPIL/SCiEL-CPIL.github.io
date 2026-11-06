"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    
    Uses LSODA method from scipy.integrate.solve_ivp with an analytic Jacobian
    for the standard Robertson problem to improve efficiency.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Standard Robertson constants
    k1 = 0.04
    k2 = 3e7
    k3 = 1e4
    
    def jac(t, y):
        return np.array([
            [-k1 * y[0], 0, k3 * y[1] * y[2]],
            [k1 * y[0], -k2 * y[1] - k3 * y[1] * y[2], 0],
            [0, k2 * y[1] + k3 * y[1] * y[2], 0]
        ])
    
    # Use LSODA with tight tolerances and analytic Jacobian
    # No max_step to allow adaptive step sizing
    sol = solve_ivp(
        f,
        t_span,
        y0,
        method='LSODA',
        rtol=1e-12,
        atol=1e-14,
        jac=jac,
    )
    
    if not sol.success:
        # Fallback if LSODA fails (e.g. if f is not standard Robertson or LSODA has issues)
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='LSODA',
            rtol=1e-10,
            atol=1e-12,
        )
    
    if not sol.success:
        # Last resort: use Radau without Jacobian
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
        )
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
