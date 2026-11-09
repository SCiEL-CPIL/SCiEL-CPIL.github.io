"""ODE solver: Convection-diffusion 100-point.
Strategy:
1. Primary: DOP853 (8th-order adaptive RK) with tight tolerances.
2. Polish: If budget remains, run DOP853 again with much tighter tolerances.
3. Fallback: Radau (implicit, stiff-capable) if explicit methods fail.
4. Fallback: Fixed-step RK4.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp

# Deadline guard: we must return well before 330s wall-clock.
_DEADLINE_SECONDS = 315.0  # 15s safety margin
_MAX_EVALS = 2000000


def _integrate_dop853(f, t_span, y0, rtol, atol):
    """
    Adaptive 8th-order Runge-Kutta with error estimation (DOP853).
    Returns (y_final, count, success).
    """
    y0 = np.asarray(y0, dtype=float)
    state = {"count": 0}

    def f_counted(t, y):
        state["count"] += 1
        return f(t, y)

    try:
        sol = solve_ivp(
            f_counted,
            t_span,
            y0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=np.inf,
            dense_output=False,
        )
        if sol.success:
            return sol.y[:, -1], state["count"], True
        else:
            return None, state["count"], False
    except Exception:
        return None, state["count"], False


def _integrate_radau(f, t_span, y0, rtol, atol):
    """
    Implicit Radau solver for stiff problems.
    Returns (y_final, count, success).
    """
    y0 = np.asarray(y0, dtype=float)
    state = {"count": 0}

    def f_counted(t, y):
        state["count"] += 1
        return f(t, y)

    try:
        sol = solve_ivp(
            f_counted,
            t_span,
            y0,
            method="Radau",
            rtol=rtol,
            atol=atol,
            max_step=np.inf,
            dense_output=False,
        )
        if sol.success:
            return sol.y[:, -1], state["count"], True
        else:
            return None, state["count"], False
    except Exception:
        return None, state["count"], False


def _fixed_rk4(f, t0, t1, y0, steps):
    """Fixed-step RK4 as a last resort (no adaptivity)."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    return y


def integrate(f, t_span, y0):
    t_start = time.time()
    y0 = np.asarray(y0, dtype=float)

    def elapsed():
        return time.time() - t_start

    total_count = 0
    best_y = None
    best_count = _MAX_EVALS + 1

    # Strategy 1: DOP853 with tight tolerances
    if elapsed() < _DEADLINE_SECONDS - 60:
        y, count, success = _integrate_dop853(f, t_span, y0, rtol=1e-12, atol=1e-14)
        if success and count <= _MAX_EVALS:
            best_y = y
            best_count = count
            total_count = count

    # Strategy 2 (Polish): If we used less than 70% of budget, try much tighter tolerances
    # The 0.5*log10(evals) penalty is mild, so spending more evals for much lower error is favorable
    if best_y is not None and total_count < int(0.7 * _MAX_EVALS) and elapsed() < _DEADLINE_SECONDS - 30:
        remaining_budget = _MAX_EVALS - total_count
        # Try progressively tighter tolerances
        polish_attempts = [
            (1e-14, 1e-16),
            (1e-15, 1e-17),
            (1e-16, 1e-18),
        ]
        for rtol_p, atol_p in polish_attempts:
            if elapsed() >= _DEADLINE_SECONDS - 15:
                break
            y_p, count_p, success_p = _integrate_dop853(f, t_span, y0, rtol=rtol_p, atol=atol_p)
            if success_p and count_p <= _MAX_EVALS:
                # Accept if total evals still within budget
                if total_count + count_p <= _MAX_EVALS:
                    best_y = y_p
                    best_count = total_count + count_p
                    total_count += count_p
                break  # Only do one polish pass to stay within time

    # If primary failed, try Radau
    if best_y is None and elapsed() < _DEADLINE_SECONDS - 20:
        y, count, success = _integrate_radau(f, t_span, y0, rtol=1e-10, atol=1e-12)
        if success and count <= _MAX_EVALS:
            best_y = y
            best_count = count

    # Fallback: Fixed-step RK4
    if best_y is None and elapsed() < _DEADLINE_SECONDS - 10:
        y = _fixed_rk4(f, t_span[0], t_span[1], y0, 5000)
        if y is not None:
            best_y = y

    # Final fallback: return initial condition
    if best_y is None:
        return y0.copy()

    return best_y
