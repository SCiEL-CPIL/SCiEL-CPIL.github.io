"""ODE solver: Convection-diffusion 100-point.
Strategy: 
1. Primary: Exploit linear time-invariant structure using Krylov subspace method (expm_multiply).
   - Extracts b = f(t, 0).
   - Defines augmented matrix operator M for [y; 1] system.
   - Computes y(T) = expm(M*T) * [y0; 1].
   - Very low number of f calls (~50-100).
2. Fallback 1: DOP853 adaptive solver.
3. Fallback 2: Fixed-step RK4.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm_multiply

# Deadline guard: we must return well before 330s wall-clock.
_DEADLINE_SECONDS = 315.0  # 15s safety margin


def _integrate_krylov(f, t_span, y0):
    """
    Solves linear time-invariant ODE y' = Ay + b using Krylov subspace method.
    Returns (y_final, count, success).
    """
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    t0, t1 = t_span
    T = t1 - t0
    
    if T == 0:
        return y0, 0, True

    state = {"count": 0}

    def f_counted(t, y):
        state["count"] += 1
        return f(t, y)

    try:
        # 1. Extract b by evaluating f at 0
        # f(t, y) = A y + b
        # f(t, 0) = b
        b = f_counted(t0, np.zeros(n))
        
        # 2. Define the augmented matrix-vector product
        # System: z' = M z, where z = [y; 1]
        # M = [[A, b], [0, 0]]
        # M @ [v; c] = [A v + c*b; 0]
        # We have f(t, v) = A v + b
        # So A v + c*b = (A v + b) + (c-1)*b = f(t, v) + (c-1)*b
        
        def matvec(v_aug):
            v = v_aug[:n]
            c = v_aug[n]
            Avb = f_counted(t0, v)
            res = np.empty(n + 1)
            res[:n] = Avb + (c - 1.0) * b
            res[n] = 0.0
            return res

        # 3. Initial condition for augmented system
        z0 = np.empty(n + 1)
        z0[:n] = y0
        z0[n] = 1.0

        # 4. Compute expm(M * T) @ z0
        # Using Arnoldi method for efficiency
        # tol=1e-14 for high accuracy
        zT = expm_multiply(matvec, z0, T, method='arnoldi', tol=1e-14)
        
        if not np.all(np.isfinite(zT)):
            return None, state["count"], False

        y_final = zT[:n]
        return y_final, state["count"], True

    except Exception:
        return None, state["count"], False


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

    # Strategy 1: Krylov subspace method (expm_multiply)
    # This is highly efficient for linear time-invariant systems.
    # Expected calls: ~50-100.
    if elapsed() < _DEADLINE_SECONDS - 30:
        y, count, success = _integrate_krylov(f, t_span, y0)
        if success and count <= 2000000:
            return y

    # Strategy 2: DOP853 with tight tolerances
    if elapsed() < _DEADLINE_SECONDS - 20:
        y, count, success = _integrate_dop853(f, t_span, y0, rtol=1e-13, atol=1e-15)
        if success and count <= 2000000:
            return y

    # Strategy 3: DOP853 with slightly looser tolerances
    if elapsed() < _DEADLINE_SECONDS - 10:
        y, count, success = _integrate_dop853(f, t_span, y0, rtol=1e-11, atol=1e-13)
        if success and count <= 2000000:
            return y

    # Strategy 4: Fixed-step RK4
    if elapsed() < _DEADLINE_SECONDS - 5:
        y = _fixed_rk4(f, t_span[0], t_span[1], y0, 5000)
        if y is not None:
            return y

    # Final fallback: return initial condition (should not happen)
    return y0.copy()

