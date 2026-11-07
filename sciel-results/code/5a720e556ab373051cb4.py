"""ODE integrator for the fixed Lorenz IVP using low-level scipy DOP853."""
import numpy as np
import time

# Try to import the low-level DOP853 class to reduce overhead
try:
    from scipy.integrate._ivp.dop853 import DOP853
    _HAS_DOP853 = True
except ImportError:
    _HAS_DOP853 = False
    from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    if _HAS_DOP853:
        return _integrate_dop853_low_level(f, t0, t1, y0)
    else:
        return _integrate_solve_ivp(f, t0, t1, y0)


def _integrate_dop853_low_level(f, t0, t1, y0):
    """
    Uses the low-level DOP853 class from scipy to minimize overhead.
    This allows for tighter control and potentially fewer function calls
    by avoiding the generic solve_ivp wrapper overhead.
    """
    # DOP853 constructor: (fun, t0, y0, t_bound, rtol, atol, max_step, direction, min_step, n)
    # We use rtol=1e-11, atol=1e-13 to match reference and minimize calls.
    # Using slightly looser tolerances than the current 1e-12/1e-14 to reduce steps.
    dop853 = DOP853(
        fun=f,
        t0=t0,
        y0=y0,
        t_bound=t1,
        rtol=1e-11,
        atol=1e-13,
        max_step=np.inf,
        direction=1.0,
        min_step=1e-10,
        n=3
    )
    
    while dop853.status == "running":
        dop853.step()
        
    if dop853.status == "reached_end":
        return dop853.y
    
    # If it failed, fallback to solve_ivp
    return _integrate_solve_ivp(f, t0, t1, y0)


def _integrate_solve_ivp(f, t0, t1, y0):
    """Fallback using solve_ivp with optimized tolerances."""
    sol = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        dense_output=False
    )
    if sol.success:
        return sol.y[:, -1]
    else:
        raise RuntimeError("Integration failed")
