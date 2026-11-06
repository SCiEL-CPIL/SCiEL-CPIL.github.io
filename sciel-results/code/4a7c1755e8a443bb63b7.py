"""Improved ODE solver for Robertson problem using DOP853 + Radau finishing step."""
import numpy as np
from scipy.integrate import solve_ivp
import warnings


def _dop853_solve(f, t0, t1, y0, rtol, atol, max_step):
    """Solve IVP using DOP853 with specified tolerances."""
    sol = solve_ivp(
        f, (t0, t1), y0,
        method='DOP853',
        rtol=rtol, atol=atol,
        max_step=max_step,
        dense_output=False
    )
    return sol.y[:, -1]


def _radau_solve(f, t0, t1, y0, rtol, atol, max_step):
    """Solve IVP using Radau with specified tolerances."""
    sol = solve_ivp(
        f, (t0, t1), y0,
        method='Radau',
        rtol=rtol, atol=atol,
        max_step=max_step,
        jac='2-point',
        dense_output=False
    )
    return sol.y[:, -1]


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy:
    1. Use DOP853 for the bulk of the integration [t0, 39.9] with relaxed tolerances
       to handle the fast transient phase efficiently.
    2. Use Radau for the final segment [39.9, t1] with tight tolerances to capture
       the slow manifold convergence accurately with minimal steps.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Phase 1: DOP853 from t0 to t1 - 0.1 (or t1 if range is small)
    t_mid = t1 - 0.1
    if t_mid <= t0:
        # If the interval is too short, just use Radau for the whole thing
        return _radau_solve(f, t0, t1, y0, rtol=1e-12, atol=1e-14, max_step=(t1 - t0))
    
    # Use relaxed tolerances for the transient phase
    y_mid = _dop853_solve(f, t0, t_mid, y0, rtol=1e-8, atol=1e-10, max_step=0.5)
    
    # Phase 2: Radau from t_mid to t1 with tight tolerances
    y_end = _radau_solve(f, t_mid, t1, y_mid, rtol=1e-13, atol=1e-15, max_step=0.1)
    
    return y_end
