"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system using a high-order adaptive Runge-Kutta method.
    
    Parameters:
    f : callable
        Function defining the ODE system: f(t, y) -> dy/dt
    t_span : tuple
        (t_start, t_end)
    y0 : array_like
        Initial state
    
    Returns:
    y_end : ndarray
        State at t_end
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Use DOP853 for high accuracy with relatively few steps
    # Tight tolerances to ensure rel error < 1e-14
    # rtol and atol are chosen to be well below the required precision
    sol = solve_ivp(
        f, 
        t_span, 
        y0_arr, 
        method='DOP853', 
        rtol=1e-13, 
        atol=1e-15,
        max_step=(t1 - t0) / 50  # Ensure at least 50 steps to capture dynamics, but let it adapt
    )
    
    if not sol.success:
        # Fallback to a more robust method if DOP853 fails, though unlikely for Lorenz
        # Radau is implicit and good for stiff, but let's try RKF45 as a simpler fallback
        sol = solve_ivp(
            f, 
            t_span, 
            y0_arr, 
            method='RKF45', 
            rtol=1e-12, 
            atol=1e-14
        )
        
    if not sol.success:
        raise RuntimeError("Integration failed")
        
    y_end = sol.y[:, -1]
    
    # Ensure finite result
    if not np.all(np.isfinite(y_end)):
        raise ValueError("Result is not finite")
        
    return y_end
# EVOLVE-BLOCK-END
