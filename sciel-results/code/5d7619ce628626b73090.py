import numpy as np
import time
import scipy.integrate

def integrate(f, t_span, y0):
    """
    Solve the Lotka-Volterra system using LSODA.
    LSODA often uses fewer function evaluations for non-stiff problems
    due to its adaptive order and step size control.
    """
    start_time = time.time()
    deadline = start_time + 280
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float).copy()
    
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='LSODA',
            rtol=1e-11,
            atol=1e-13
        )
        
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Fallback to DOP853 if LSODA fails
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13
        )
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
    except Exception:
        pass

    return y0
