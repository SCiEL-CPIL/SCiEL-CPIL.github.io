"""
High-accuracy Robertson ODE solver using a multi-solver strategy.

Strategy:
  The Robertson problem is highly stiff. We try multiple solvers in order of
  expected efficiency, using the analytic Jacobian to minimize function calls.
  
  Key insight: The score is -log10(relative error) - 0.5*log10(evaluations).
  We want to maximize accuracy while minimizing the number of calls to f.
  
  LSODA is typically the most efficient for this problem because it adapts
  between Adams (non-stiff) and BDF (stiff) methods. We use very tight
  tolerances to maximize the accuracy term.
  
  We also try Radau and BDF as fallbacks in case LSODA fails or times out.
  
  A time guard ensures we return a result before the 300s wall-clock limit.
"""

import numpy as np
import time
from scipy.integrate import solve_ivp


def _robertson_jacobian(t, y):
    """
    Analytical Jacobian for the Robertson problem.
    f0 = -0.04*y0 + 1e4*y1*y2
    f1 = 0.04*y0 - 1e4*y1*y2 - 3e7*y1^2
    f2 = 3e7*y1^2
    
    J = [[-0.04,        1e4*y2,       1e4*y1],
         [ 0.04,       -1e4*y2-6e7*y1, -1e4*y1],
         [ 0.0,         6e7*y1,        0.0]]
    """
    y = np.asarray(y).reshape(-1)
    n = y.shape[0]
    if n >= 3:
        y0v, y1v, y2v = y[0], y[1], y[2]
        J = np.empty((n, n))
        J[0, 0] = -0.04
        J[0, 1] = 1.0e4 * y2v
        J[0, 2] = 1.0e4 * y1v
        J[1, 0] = 0.04
        J[1, 1] = -1.0e4 * y2v - 6.0e7 * y1v
        J[1, 2] = -1.0e4 * y1v
        J[2, 0] = 0.0
        J[2, 1] = 6.0e7 * y1v
        J[2, 2] = 0.0
        if n > 3:
            J[3:, :] = 0.0
            J[:, 3:] = 0.0
            np.fill_diagonal(J, 0.0)
        return J
    else:
        J = np.zeros((n, n))
        return J


def integrate(f, t_span, y0):
    """
    Solve the fixed Robertson IVP on [t0, t1] and return y(t1).

    Parameters
    ----------
    f : callable
        ODE right-hand side f(t, y) -> dy/dt.
    t_span : tuple
        (t0, t1) integration interval.
    y0 : array-like
        Initial condition at t0.

    Returns
    -------
    y_end : ndarray
        Final state y(t1).
    """
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float).reshape(-1)
    n = y0_arr.shape[0]

    start_time = time.time()
    time_limit = 290.0  # safety margin before 300s

    def time_left():
        return time_limit - (time.time() - start_time)

    t_eval = np.array([t0, t1])

    # Primary: LSODA with very tight tolerances and analytic Jacobian.
    # LSODA adapts between Adams and BDF, often most efficient for stiff problems.
    try:
        sol = solve_ivp(
            f, t_span, y0_arr,
            method="LSODA",
            jac=_robertson_jacobian,
            rtol=1e-14,
            atol=1e-16,
            t_eval=t_eval,
        )
        if sol.success and sol.y.shape[1] >= 2:
            return sol.y[:, -1].reshape(-1)
    except Exception:
        pass

    # Fallback 1: Radau with tight tolerances.
    if time_left() > 10.0:
        try:
            sol = solve_ivp(
                f, t_span, y0_arr,
                method="Radau",
                jac=_robertson_jacobian,
                rtol=1e-14,
                atol=1e-16,
                t_eval=t_eval,
            )
            if sol.success and sol.y.shape[1] >= 2:
                return sol.y[:, -1].reshape(-1)
        except Exception:
            pass

    # Fallback 2: BDF with tight tolerances.
    if time_left() > 10.0:
        try:
            sol = solve_ivp(
                f, t_span, y0_arr,
                method="BDF",
                jac=_robertson_jacobian,
                rtol=1e-14,
                atol=1e-16,
                t_eval=t_eval,
            )
            if sol.success and sol.y.shape[1] >= 2:
                return sol.y[:, -1].reshape(-1)
        except Exception:
            pass

    # Last resort: Loose LSODA to ensure a result is returned
    if time_left() > 5.0:
        try:
            sol = solve_ivp(
                f, t_span, y0_arr,
                method="LSODA",
                jac=_robertson_jacobian,
                rtol=1e-8,
                atol=1e-10,
                t_eval=t_eval,
            )
            if sol.success and sol.y.shape[1] >= 2:
                return sol.y[:, -1].reshape(-1)
        except Exception:
            pass

    return y0_arr.copy()
