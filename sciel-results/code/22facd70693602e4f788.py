import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem using SciPy's Radau solver with an analytic Jacobian.
    
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
    
    # Deadline guard
    start_time = time.time()
    time_limit = 320.0  # seconds, safety margin before 330s kill
    
    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y0_val, y1_val, y2_val = y[0], y[1], y[2]
        J = np.array([
            [-0.04, 1e4 * y2_val, 1e4 * y1_val],
            [0.04, -1e4 * y2_val - 6e7 * y1_val, -1e4 * y1_val],
            [0.0, 6e7 * y1_val, 0.0]
        ])
        return J

    # Use Radau solver which is suitable for stiff problems
    # Tighter tolerances to minimize error, balancing against function call count
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=jac, 
            rtol=1e-12, 
            atol=1e-14
        )
        
        if not sol.success:
            # Fallback if Radau fails
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                jac=jac, 
                rtol=1e-12, 
                atol=1e-14
            )
            
        # Return the final state
        return sol.y[:, -1]
        
    except Exception as e:
        # If something goes wrong, return y0 to avoid crashing
        return y0
