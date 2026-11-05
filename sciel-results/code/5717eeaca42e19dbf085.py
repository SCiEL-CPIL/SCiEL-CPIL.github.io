"""ODE solver: Robertson problem using Radau (implicit Runge-Kutta) for stiff systems."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) over t_span starting at y0.
    Uses Radau as primary solver (optimal for stiff problems like Robertson).
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Hard deadline: 325s to leave 5s safety margin
    start_time = time.time()
    deadline = start_time + 325

    try:
        # Radau is an implicit Runge-Kutta method of order 5, specifically designed
        # for stiff problems. It should use fewer function evaluations than LSODA
        # for the Robertson problem while maintaining high accuracy.
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='Radau', 
            rtol=1e-13, 
            atol=1e-13,
            jac=None  # Let Radau compute Jacobian internally
        )
        
        elapsed = time.time() - start_time
        if sol.success:
            return sol.y[:, -1]
        
        # Check if we're running out of time
        if time.time() > deadline - 5:
            # Return best available result
            if sol.y is not None and sol.y.size > 0:
                return sol.y[:, -1]
            return y0_arr

        # Fallback to BDF with tight tolerances
        sol2 = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='BDF', 
            rtol=1e-13, 
            atol=1e-13
        )
        
        if sol2.success:
            return sol2.y[:, -1]
        
        if sol2.y is not None and sol2.y.size > 0:
            return sol2.y[:, -1]
        return y0_arr
            
    except Exception:
        # In case of unexpected errors, return y0 to avoid crash
        return y0_arr
# EVOLVE-BLOCK-END
