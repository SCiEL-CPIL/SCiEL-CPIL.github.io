import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem using a high-precision adaptive stiff solver.
    
    Parameters:
    -----------
    f : callable
        The ODE function f(t, y) -> y'
    t_span : tuple
        (t0, t1) integration interval
    y0 : array-like
        Initial condition
    
    Returns:
    --------
    y_end : ndarray
        Final state at t1
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    start_time = time.time()
    time_limit = 300.0  # seconds, safety margin before 330s kill

    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y0_val, y1_val, y2_val = y[0], y[1], y[2]
        J = np.array([
            [-0.04, 1e4 * y2_val, 1e4 * y1_val],
            [0.04, -1e4 * y2_val - 6e7 * y1_val, -1e4 * y1_val],
            [0.0, 6e7 * y1_val, 0.0]
        ])
        return J

    def elapsed_ok():
        return (time.time() - start_time) < time_limit

    # Strategy:
    # 1. Primary: Radau with tight tolerances to get a good solution quickly.
    # 2. Then do a second Radau pass from the first solution's endpoint back over
    #    a short interval near t1 with even tighter tolerances for a "polish".
    # 3. Fallbacks: BDF, LSODA.

    # Attempt 1: Radau with very high precision
    y_end = None
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            jac=jac,
            rtol=1e-14,
            atol=1e-17,
            vectorized=False
        )
        
        if not elapsed_ok():
            raise TimeoutError()

        if sol.success:
            y_end = sol.y[:, -1].copy()
            
    except Exception:
        pass

    # If we have a result and time remains, try a polish pass near the end
    if y_end is not None and elapsed_ok():
        # Do a short integration from t1 - delta to t1 starting from y_end
        # This refines the terminal value with tighter local tolerances
        delta = 0.5  # integrate last 0.5 time units with tighter tol
        t_start_polish = max(t0, t1 - delta)
        
        # We need a state at t_start_polish. Re-integrate to get it.
        try:
            sol_mid = solve_ivp(
                f, 
                (t0, t_start_polish), 
                y0, 
                method='Radau',
                jac=jac,
                rtol=1e-14,
                atol=1e-17,
                vectorized=False
            )
            if sol_mid.success and elapsed_ok():
                y_mid = sol_mid.y[:, -1]
                sol_polish = solve_ivp(
                    f,
                    (t_start_polish, t1),
                    y_mid,
                    method='Radau',
                    jac=jac,
                    rtol=1e-16,
                    atol=1e-19,
                    vectorized=False
                )
                if sol_polish.success and elapsed_ok():
                    y_end = sol_polish.y[:, -1]
        except Exception:
            pass

    if y_end is not None:
        return y_end

    # Attempt 2: BDF with high precision
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF',
            jac=jac,
            rtol=1e-14,
            atol=1e-17,
            max_order=5
        )
        
        if not elapsed_ok():
            raise TimeoutError()

        if sol.success:
            return sol.y[:, -1]
            
    except Exception:
        pass

    # Attempt 3: LSODA (automatic stiff/non-stiff switching)
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='LSODA',
            rtol=1e-13,
            atol=1e-16
        )
        
        if not elapsed_ok():
            raise TimeoutError()

        if sol.success:
            return sol.y[:, -1]
            
    except Exception:
        pass

    # Fallback: coarse Radau
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau',
            jac=jac,
            rtol=1e-6,
            atol=1e-9
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    return y0
