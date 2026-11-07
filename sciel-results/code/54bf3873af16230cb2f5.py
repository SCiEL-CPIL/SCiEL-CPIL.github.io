"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrates the ODE defined by f over t_span starting from y0.
    
    This implementation uses scipy's DOP853 solver with tight tolerances
    to match the reference solution as closely as possible, minimizing
    relative error.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use DOP853 with the exact tolerances specified in the task contract
    # to ensure the trajectory matches y_ref as closely as floating point allows.
    sol = solve_ivp(
        f, 
        t_span, 
        y0, 
        method='DOP853', 
        rtol=1e-11, 
        atol=1e-13,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback if for some reason it fails, though DOP853 is very robust
        # Try RK45 with tighter tolerances
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-11, 
            atol=1e-13,
            dense_output=False
        )
        
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
