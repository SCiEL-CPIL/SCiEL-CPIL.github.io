"""Solve the fixed 15-dim stiff linear IVP y' = f(t, y) on [0,1].

Strategy: for a linear time-invariant system y' = A y, the exact terminal
state is y(T) = expm(A*(T-t0)) @ y0.  We extract A by probing f along the
15 coordinate directions (15 calls), verify linearity/time-invariance using
the values we already computed (no extra calls), and fall back to a stiff
implicit integrator only if the problem is not linear.  This yields
machine-precision accuracy at ~15 function evaluations, which is near-optimal
for the score -log10(rel_err) - 0.5*log10(f_calls).
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import expm

_DEADLINE = 330.0
_T0_LIMIT = 300.0  # hard wall-clock guard with margin


def _elapsed_guard():
    return time.perf_counter()


def _extract_A(f, t0, y0):
    """Probe f along coordinate directions to build A.  15 calls.

    For a linear system f(t, y) = A(t) y (and A independent of t), the
    column j of A is f(t, e_j) (with e_j the unit vector) because f(t, e_j) =
    A e_j.  We use the coordinate basis so no dependence on y0 scale is needed.
    """
    n = len(y0)
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        col = f(t0, e)
        A[:, j] = np.asarray(col, dtype=float)
    return A


def _is_linear_time_invariant(f, t0, t1, y0, A, tol=1e-7):
    """Cheap linearity / time-invariance check using A.  2 extra calls.

    Checks f(t, c*y0) == c * A @ y0  (linearity in y) and
    f(t1, y0) == A @ y0               (time invariance).
    """
    n = len(y0)
    c = 1.7  # arbitrary non-unit scalar
    y_test = c * y0
    pred = c * (A @ y0)
    got = np.asarray(f(t0, y_test), dtype=float)
    scale = max(1.0, np.linalg.norm(pred))
    if np.linalg.norm(got - pred) > tol * scale:
        return False
    got_t1 = np.asarray(f(t1, y0), dtype=float)
    pred_t1 = A @ y0
    scale2 = max(1.0, np.linalg.norm(pred_t1))
    if np.linalg.norm(got_t1 - pred_t1) > tol * scale2:
        return False
    return True


def _backward_euler(f, t0, t1, y0, steps):
    """Implicit Euler with finite-difference Jacobian + Newton (stiff fallback)."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    n = len(y)
    eye = np.eye(n)
    for _ in range(steps):
        z = y.copy()
        for _newton in range(6):
            fz = f(t + h, z)
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-7 * max(1.0, abs(z[j]))
                J[:, j] = (f(t + h, z + e) - fz) / e[j]
            rhs = -(z - y - h * fz)
            try:
                step = np.linalg.solve(eye - h * J, rhs)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(eye - h * J, rhs, rcond=None)[0]
            z = z + step
            if np.linalg.norm(step) <= 1e-11 * (1 + np.linalg.norm(z)):
                break
        y, t = z, t + h
    return y


def _rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
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
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    start = _elapsed_guard()

    # --- Primary: matrix exponential for the linear time-invariant case. ---
    # 15 calls to build A.
    A = _extract_A(f, t0, y0)
    if not np.all(np.isfinite(A)):
        # Non-finite probe: fall through to a robust integrator.
        pass
    else:
        # 2 calls to verify linearity + time invariance.
        if _is_linear_time_invariant(f, t0, t1, y0, A):
            M = expm(A * (t1 - t0))
            y = M @ y0
            if np.all(np.isfinite(y)):
                # High-precision polish: recompute with a Pade-based expm is
                # already machine precision; just guard the deadline.
                if _elapsed_guard() - start < _T0_LIMIT:
                    return y
                return y
    # --- Fallback: stiff integrators (only reached if not linear). ---
    if _elapsed_guard() - start > _T0_LIMIT:
        # Return best-effort: a single backward-Euler step is risky; use RK4
        # with a modest step count as a finite, safe answer.
        y = _rk4(f, t0, t1, y0, 2000)
        return y if y is not None else y0.copy()
    y = _backward_euler(f, t0, t1, y0, 2000)
    if y is not None and np.all(np.isfinite(y)):
        return y
    y = _rk4(f, t0, t1, y0, 2000)
    return y if y is not None else y0.copy()
# EVOLVE-BLOCK-END
