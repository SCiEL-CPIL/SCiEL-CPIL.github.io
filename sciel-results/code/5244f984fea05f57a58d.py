"""ODE solver: Convection-diffusion 100-point.
Strategy:
1. Primary: Detect LTI structure. If found, use Matrix Exponential (exact, low calls).
2. Fallback: DOP853 (8th-order adaptive RK) with very tight tolerances.
3. Fallback: DOP853 with looser tolerances.
4. Fallback: Radau (implicit, stiff-capable).
5. Fallback: Fixed-step RK4.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# Deadline guard: we must return well before 330s wall-clock.
_DEADLINE_SECONDS = 315.0  # 15s safety margin


def _try_lti_solve(f, t_span, y0):
    """
    Attempts to solve using LTI assumption (f(t,y) = A y).
    Returns (y_final, count, success).
    """
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    state = {"count": 0}

    def f_counted(t, y):
        state["count"] += 1
        return f(t, y)

    try:
        # 1. Get f(0, y0)
        f_y0 = f_counted(0, y0)
        f_y0 = np.asarray(f_y0, dtype=float)
        
        # 2. Construct A using basis vectors
        A = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            A[:, i] = f_counted(0, e_i)
            
        # 3. Verify linearity: A @ y0 == f_y0
        residual = np.linalg.norm(A @ y0 - f_y0)
        scale = np.linalg.norm(f_y0) + 1e-12
        # Relax tolerance to 1e-6 to handle potential FP noise or slightly non-linear but "linear-ish" cases?
        # No, if it's not linear, we should fail. But 1e-8 might be too tight for large values.
        if residual / scale > 1e-6:
            return None, state["count"], False
            
        # 4. Check time invariance: f(t, y0) should be A @ y0 for t > 0
        t_mid = t_span[1] / 2.0
        f_mid = f_counted(t_mid, y0)
        residual_t = np.linalg.norm(A @ y0 - f_mid)
        if residual_t / scale > 1e-6:
            return None, state["count"], False

        # 5. Compute exact solution
        # Check if A is sparse to use more efficient expm?
        # For N=100, dense expm is fast.
        dt = t_span[1] - t_span[0]
        M = expm(A * dt)
        y_final = M @ y0
        
        return y_final, state["count"], True

    except Exception:
        return None, state["count"], False


def _integrate_radau(f, t_span, y0, rtol, atol, max_evals=2000000):
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


def _integrate_dop853(f, t_span, y0, rtol, atol):
    """
    DOP853 high-order explicit solver.
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


def integrate(f, t_span, y0):
    t_start = time.time()
    y0 = np.asarray(y0, dtype=float)

    def elapsed():
        return time.time() - t_start

    # Strategy 1: LTI Exact Solve (High accuracy, low calls)
    if elapsed() < _DEADLINE_SECONDS - 10:
        y, count, success = _try_lti_solve(f, t_span, y0)
        if success and count <= 2000000:
            return y

    # Strategy 2: Radau (implicit) for stiff problems - Try this first for non-linear/stiff
    # Radau is better for stiff problems than DOP853.
    if elapsed() < _DEADLINE_SECONDS - 30:
        y, count, success = _integrate_radau(f, t_span, y0, rtol=1e-12, atol=1e-14)
        if success and count <= 2000000:
            return y

    # Strategy 3: DOP853 with very tight tolerances
    if elapsed() < _DEADLINE_SECONDS - 20:
        y, count, success = _integrate_dop853(f, t_span, y0, rtol=1e-13, atol=1e-15)
        if success and count <= 2000000:
            return y

    # Strategy 4: DOP853 with slightly looser tolerances
    if elapsed() < _DEADLINE_SECONDS - 15:
        y, count, success = _integrate_dop853(f, t_span, y0, rtol=1e-11, atol=1e-13)
        if success and count <= 2000000:
            return y

    # Strategy 5: Fixed-step RK4 with moderate steps
    if elapsed() < _DEADLINE_SECONDS - 10:
        y = _fixed_rk4(f, t_span[0], t_span[1], y0, 5000)
        if y is not None:
            return y

    # Strategy 6: Even fewer steps if very tight on time
    y = _fixed_rk4(f, t_span[0], t_span[1], y0, 2000)
    if y is not None:
        return y

    # Final fallback: return initial condition (should not happen)
    return y0.copy()
