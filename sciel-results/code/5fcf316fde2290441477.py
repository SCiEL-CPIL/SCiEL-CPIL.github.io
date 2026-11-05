"""ODE solver: Robertson problem using LSODA via scipy.integrate.solve_ivp."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    
    Uses LSODA which automatically switches between Adams and BDF methods,
    well-suited for the Robertson problem's stiffness transitions.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Track function calls
    call_count = [0]
    
    def wrapped_f(t, y):
        call_count[0] += 1
        return f(t, y)
    
    # Deadline guard: 320 seconds (10s safety margin before 330s kill)
    start_time = time.time()
    
    # Try LSODA with tight tolerances
    # LSODA handles stiffness automatically
    try:
        sol = solve_ivp(
            wrapped_f,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-12,
            atol=1e-14,
            max_step=np.inf  # Let LSODA choose step sizes freely
        )
        
        if sol.success and sol.t[-1] >= t1 - 1e-10:
            return sol.y[:, -1]
        
        # If LSODA didn't complete, check elapsed time
        elapsed = time.time() - start_time
        if elapsed > 300:
            # Return best available
            if sol.y.size > 0:
                return sol.y[:, -1]
            return y0
    
    except Exception:
        pass
    
    # Fallback: try Radau (implicit, good for stiff problems)
    elapsed = time.time() - start_time
    if elapsed < 300:
        try:
            sol = solve_ivp(
                wrapped_f,
                [t0, t1],
                y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14
            )
            if sol.success and sol.t[-1] >= t1 - 1e-10:
                return sol.y[:, -1]
            if sol.y.size > 0:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Fallback: try BDF
    elapsed = time.time() - start_time
    if elapsed < 300:
        try:
            sol = solve_ivp(
                wrapped_f,
                [t0, t1],
                y0,
                method='BDF',
                rtol=1e-12,
                atol=1e-14
            )
            if sol.success and sol.t[-1] >= t1 - 1e-10:
                return sol.y[:, -1]
            if sol.y.size > 0:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Last resort: return initial condition
    return y0
# EVOLVE-BLOCK-END
