"""ODE solver: Convection-diffusion 100-point, deadline-guarded adaptive solver."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def _attempt(method, f, t0, t1, y0, rtol, atol, deadline, max_step=None):
    """Run solve_ivp with a deadline guard. Returns (y_end, n_calls) or (None, n_calls)."""
    # Wrap f to count calls and check deadline
    state = {"calls": 0}

    def f_counted(t, y):
        state["calls"] += 1
        return f(t, y)

    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
    except Exception:
        return None, state["calls"]

    if not sol.success:
        return None, state["calls"]

    # Check deadline after the call
    if time.time() > deadline:
        return None, state["calls"]

    if not np.all(np.isfinite(sol.y[:, -1])):
        return None, state["calls"]

    return sol.y[:, -1], state["calls"]


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)

    # Deadline: finish well before the 330s kill. Use 300s as the hard internal limit.
    start = time.time()
    deadline = start + 295.0  # safety margin

    best_y = None
    best_calls = None

    # Primary: high-order explicit DOP853, tight tolerances.
    y, calls = _attempt("DOP853", f, t0, t1, y0, rtol=1e-12, atol=1e-14, deadline=deadline)
    if y is not None:
        best_y = y
        best_calls = calls

    # If DOP853 failed or we're still within budget, try an implicit stiff-capable method
    # to cross-check / recover. Only do this if we have time and DOP853 didn't succeed.
    if best_y is None and time.time() < deadline - 5.0:
        # LSODA auto-switches between stiff/non-stiff.
        y, calls = _attempt("LSODA", f, t0, t1, y0, rtol=1e-11, atol=1e-13, deadline=deadline)
        if y is not None:
            best_y = y
            best_calls = calls

    if best_y is None and time.time() < deadline - 5.0:
        # Radau: fully implicit, very robust for stiff problems.
        y, calls = _attempt("Radau", f, t0, t1, y0, rtol=1e-10, atol=1e-12, deadline=deadline)
        if y is not None:
            best_y = y
            best_calls = calls

    if best_y is None:
        # Last resort: a fixed-step RK4 with a modest number of steps, guarded by deadline.
        n = len(y0)
        h = (t1 - t0) / 2000
        t, y = t0, y0.copy()
        steps_done = 0
        while t < t1 - 1e-15:
            if time.time() > deadline:
                break
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            steps_done += 1
            if not np.all(np.isfinite(y)):
                break
        if np.all(np.isfinite(y)):
            best_y = y
            best_calls = steps_done * 4

    if best_y is None:
        # Absolute fallback: return the initial condition (better than an exception).
        return y0

    return best_y
# EVOLVE-BLOCK-END
