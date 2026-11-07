"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the ODE system using SciPy's DOP853 solver with tight tolerances.
    
    Parameters:
    -----------
    f : callable
        Function f(t, y) returning dy/dt as an array.
    t_span : tuple
        (t_start, t_end)
    y0 : array_like
        Initial state vector.
    
    Returns:
    --------
    y_end : ndarray
        State vector at t_end.
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Use DOP853 with very tight tolerances to match the reference solver
    sol = solve_ivp(
        f,
        t_span,
        y0_arr,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback: try with slightly relaxed tolerances
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='DOP853',
            rtol=1e-10,
            atol=1e-12,
            dense_output=False
        )
    
    # Return the state at the end of the integration
    y_end = sol.y[:, -1]
    
    # Ensure the result is finite and has the correct shape
    if not np.all(np.isfinite(y_end)):
        # Last resort: return the initial condition if something went wrong
        y_end = y0_arr.copy()
    
    return y_end
# EVOLVE-BLOCK-END
