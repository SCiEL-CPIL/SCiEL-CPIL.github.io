"""ODE solver: Robertson problem using scipy.integrate.solve_ivp with stiff solver."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use LSODA which automatically handles stiff and non-stiff problems
    # rtol and atol set tight enough for good accuracy
    # max_step limits step size for accuracy
    result = solve_ivp(
        f,
        t_span,
        y0,
        method='LSODA',
        rtol=1e-12,
        atol=1e-14,
        max_step=0.1
    )
    
    if result.success:
        return result.y[:, -1]
    
    # Fallback: try Radau if LSODA fails
    result = solve_ivp(
        f,
        t_span,
        y0,
        method='Radau',
        rtol=1e-12,
        atol=1e-14,
        max_step=0.1
    )
    
    if result.success:
        return result.y[:, -1]
    
    # Last resort: BDF
    result = solve_ivp(
        f,
        t_span,
        y0,
        method='BDF',
        rtol=1e-10,
        atol=1e-12,
        max_step=0.5
    )
    
    if result.success:
        return result.y[:, -1]
    
    # If all fail, return last computed value or y0
    if result.y.size > 0:
        return result.y[:, -1]
    return y0
# EVOLVE-BLOCK-END
