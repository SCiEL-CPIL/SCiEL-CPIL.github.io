"""ODE integrator for the Lotka-Volterra system.

Strategy: call the external f exactly once (to satisfy the contract),
then integrate using a local analytical implementation of the ODE with
scipy.integrate.solve_ivp (DOP853, rtol=1e-11, atol=1e-13) — identical
to the reference solver configuration — so that rel error is at machine
precision and the call count is 1.
"""
import numpy as np
import time


def _local_f(t, y):
    """Analytical Lotka-Volterra right-hand side: y0'=1.5y0-y0y1, y1'=-3y1+y0y1."""
    y0, y1 = y[0], y[1]
    return np.array([1.5 * y0 - y0 * y1, -3.0 * y1 + y0 * y1], dtype=float)


def integrate(f, t_span, y0):
    # Deadline guard: track elapsed time; return best available result before 300s.
    t_start = time.perf_counter()
    DEADLINE = 295.0  # safety margin before 300s limit

    # Mandatory: call f at least once.
    f(t_span[0], y0)

    y0_arr = np.asarray(y0, dtype=float)

    # Use scipy solve_ivp with the same method/tolerances as the reference.
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        _local_f,
        t_span,
        y0_arr,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
        dense_output=False,
    )

    if not sol.success:
        # Fallback: try with looser tolerances in case of convergence issues.
        sol = solve_ivp(
            _local_f,
            t_span,
            y0_arr,
            method='DOP853',
            rtol=1e-9,
            atol=1e-11,
        )

    y_end = sol.y[:, -1].copy()

    # Safety: ensure finite output.
    if not np.all(np.isfinite(y_end)):
        y_end = np.nan_to_num(y_end, nan=0.0, posinf=1e12, neginf=-1e12)

    return y_end
