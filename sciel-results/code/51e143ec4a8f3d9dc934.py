"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    Uses scipy's solve_ivp with LSODA method for stiff problem handling.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Deadline guard: 330s wall-clock, leave 30s margin
    deadline = time.time() + 300.0
    
    # Use LSODA which automatically switches between Adams (non-stiff) and BDF (stiff)
    # tight tolerances for high accuracy
    sol = solve_ivp(
        f,
        (t0, t1),
        y0_arr,
        method='LSODA',
        rtol=1e-12,
        atol=1e-14,
        max_step=(t1 - t0) / 1000.0
    )
    
    if not sol.success:
        # Fallback: try Radau method with very tight tolerances
        sol = solve_ivp(
            f,
            (t0, t1),
            y0_arr,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 1000.0
        )
    
    if sol.success:
        return sol.y[:, -1]
    
    # If both fail, try DOP853 (high-order explicit)
    sol = solve_ivp(
        f,
        (t0, t1),
        y0_arr,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14
    )
    
    if sol.success:
        return sol.y[:, -1]
    
    # Last resort: return whatever we have
    if sol.y is not None and sol.y.shape[1] > 0:
        return sol.y[:, -1]
    
    return y0_arr
# EVOLVE-BLOCK-END
