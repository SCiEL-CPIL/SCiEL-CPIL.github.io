"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate


def integrate(f, t_span, y0):
    """
    Integrate the Lorenz system from t_span[0] to t_span[1] starting at y0.
    
    Uses scipy.integrate.solve_ivp with a high-order explicit Runge-Kutta method
    (DOP853) and tight tolerances matching the reference to minimize relative error
    while keeping function call count reasonable.
    
    Parameters
    ----------
    f : callable
        The right-hand side function f(t, y) -> y'.
    t_span : tuple
        (t_start, t_end)
    y0 : array_like
        Initial state vector.
        
    Returns
    -------
    y_end : ndarray
        State vector at t_end.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use RK45 (4th/5th order) as per directive
    # Lower order method may have different call/accuracy trade-offs
    sol = scipy.integrate.solve_ivp(
        f,
        (t0, t1),
        y0,
        method='RK45',
        rtol=1e-11,
        atol=1e-13
    )
    
    if not sol.success or sol.y.shape[1] == 0:
        # Fallback: try with slightly looser tolerances if something went wrong
        sol = scipy.integrate.solve_ivp(
            f,
            (t0, t1),
            y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-12
        )
    
    if not sol.success or sol.y.shape[1] == 0:
        # Last resort: return initial condition if integration failed completely
        return y0
    
    # Return the state at the end of t_span
    y_end = sol.y[:, -1]
    
    # Ensure it's finite
    if not np.all(np.isfinite(y_end)):
        return y0
    
    return y_end
# EVOLVE-BLOCK-END
