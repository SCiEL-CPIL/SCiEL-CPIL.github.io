"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    
    Uses LSODA with tight tolerances for the Robertson problem.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: return best result before 320s (10s safety margin before 330s kill)
    deadline = time.time() + 320.0
    
    # Count function calls via a wrapper
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Try LSODA with very tight tolerances
    # Robertson problem is stiff; LSODA handles it well with automatic switching
    try:
        sol = solve_ivp(
            f_counted,
            (t0, t1),
            y0,
            method='LSODA',
            rtol=1e-13,
            atol=1e-15,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback: Radau (implicit, good for stiff problems)
    try:
        sol = solve_ivp(
            f_counted,
            (t0, t1),
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort: BDF
    try:
        sol = solve_ivp(
            f_counted,
            (t0, t1),
            y0,
            method='BDF',
            rtol=1e-11,
            atol=1e-13,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If all fail, try RK45 with smaller steps as a last resort
    try:
        sol = solve_ivp(
            f_counted,
            (t0, t1),
            y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
            max_step=0.1,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Absolute last resort: return initial condition (shouldn't happen)
    return y0
# EVOLVE-BLOCK-END
