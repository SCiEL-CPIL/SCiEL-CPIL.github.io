"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import odeint


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] using scipy's odeint with LSODA.
    
    LSODA automatically detects stiffness and adjusts step size, which is ideal for
    the Robertson problem. This minimizes function evaluations while maintaining accuracy.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Create time points: just the start and end to get only the final state
    # odeint requires at least 2 time points
    t_eval = np.array([t0, t1])
    
    # Use LSODA with tight tolerances for high accuracy
    # rtol=1e-12 and atol=1e-14 for very accurate results
    try:
        y_end = odeint(f, y0, t_eval, method='LSODA', rtol=1e-12, atol=1e-14)
        return y_end[-1]
    except Exception:
        # Fallback to Radau if LSODA fails
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]
# EVOLVE-BLOCK-END
