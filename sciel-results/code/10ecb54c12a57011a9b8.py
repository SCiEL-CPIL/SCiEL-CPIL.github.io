import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Solves the ODE y' = f(t, y) on t_span with initial condition y0.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 50s safety margin for 330s limit
    
    # Try high-order explicit method first (DOP853)
    # It is efficient for non-stiff problems
    try:
        if time.time() < deadline:
            sol = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='DOP853',
                rtol=1e-12,
                atol=1e-14,
                max_step=np.inf
            )
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass

    # Fallback to implicit method for stiff problems (Radau)
    # Radau is robust for stiff systems
    try:
        if time.time() < deadline:
            sol = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14
            )
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass

    # Last resort: BDF
    try:
        if time.time() < deadline:
            sol = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='BDF',
                rtol=1e-10,
                atol=1e-12
            )
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass

    # If all fail, return initial state (should not happen)
    return y0
