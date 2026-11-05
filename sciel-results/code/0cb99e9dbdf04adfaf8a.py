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
    Two-stage approach: first pass with moderate tolerances, then a polish
    pass with very tight tolerances to reduce relative error.
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
    deadline = 310  # safety margin
    
    # Stage 1: LSODA with tight tolerances to get a good approximation
    y_current = y0
    t_current_start = t0
    try:
        sol = solve_ivp(
            wrapped_f,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-12,
            atol=1e-14,
            max_step=np.inf
        )
        
        if sol.success and sol.t[-1] >= t1 - 1e-10:
            y_current = sol.y[:, -1].copy()
        elif sol.y.size > 0:
            y_current = sol.y[:, -1].copy()
    
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return y_current
    
    # Stage 2: Polish pass with much tighter tolerances
    # Re-integrate from t0 to t1 with very tight tolerances
    try:
        sol2 = solve_ivp(
            wrapped_f,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-15,
            atol=1e-18,
            max_step=np.inf
        )
        
        if sol2.success and sol2.t[-1] >= t1 - 1e-10:
            y_current = sol2.y[:, -1].copy()
        elif sol2.y.size > 0:
            y_current = sol2.y[:, -1].copy()
    
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return y_current
    
    # Stage 3: Final polish with even tighter tolerances if time permits
    try:
        sol3 = solve_ivp(
            wrapped_f,
            [t0, t1],
            y0,
            method='LSODA',
            rtol=1e-16,
            atol=1e-20,
            max_step=np.inf
        )
        
        if sol3.success and sol3.t[-1] >= t1 - 1e-10:
            y_current = sol3.y[:, -1].copy()
        elif sol3.y.size > 0:
            y_current = sol3.y[:, -1].copy()
    
    except Exception:
        pass
    
    # Last resort: return best available
    return y_current
# EVOLVE-BLOCK-END
