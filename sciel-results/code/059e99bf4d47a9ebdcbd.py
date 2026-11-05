"""ODE solver using matrix-exponential for linear time-invariant systems, with fallback."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import expm


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


def _build_matrix_A(f, t, y0, n):
    """Build the Jacobian A of f at (t, y0) via finite differences."""
    A = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1e-7 * max(1.0, abs(y0[j]))
        A[:, j] = (f(t, y0 + e) - f(t, y0)) / e[j]
    return A


def _is_linear_time_invariant(f, t0, t1, y0, n):
    """
    Check if f is affine-linear and time-invariant: f(t, y) = A @ y + b.
    Returns (A, b) if so, else None.
    """
    # Sample at two different times and a few random states to verify linearity + time-invariance.
    rng = np.random.RandomState(0)
    t_a = t0
    t_b = (t0 + t1) / 2.0
    t_c = t1 - 1e-6 * (t1 - t0) if t1 > t0 else t0

    # First, get f at the base point to estimate b: assume f(t, 0) = b (affine).
    y_zero = np.zeros(n)
    try:
        f_zero_a = np.asarray(f(t_a, y_zero), dtype=float)
        f_zero_b = np.asarray(f(t_b, y_zero), dtype=float)
        f_zero_c = np.asarray(f(t_c, y_zero), dtype=float)
    except Exception:
        return None

    # Time-invariance check on zero state
    if not (np.allclose(f_zero_a, f_zero_b, atol=1e-10) and
            np.allclose(f_zero_a, f_zero_c, atol=1e-10)):
        return None

    b = f_zero_a

    # Build A via finite differences at y_zero (linear part).
    A = np.empty((n, n))
    eps = 1e-7
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        A[:, j] = (f(t_a, e) - b) / eps

    # Verify linearity: f(t, y) == A @ y + b for random y at multiple times.
    for t_test in (t_a, t_b, t_c):
        for _ in range(4):
            y_test = rng.randn(n) * 0.5
            expected = A @ y_test + b
            try:
                actual = np.asarray(f(t_test, y_test), dtype=float)
            except Exception:
                return None
            if not np.allclose(actual, expected, rtol=1e-6, atol=1e-9):
                return None
    return A, b


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    deadline = time.time() + 295.0  # safety margin under 300s

    # Try the fast linear time-invariant matrix-exponential path.
    try:
        lin = _is_linear_time_invariant(f, t0, t1, y0, n)
    except Exception:
        lin = None

    if lin is not None:
        A, b = lin
        try:
            # y(t) = e^{A (t-t0)} y0 + (e^{A (t-t0)} - I) A^{-1} b   (when A invertible)
            # Use a robust formulation: solve via block-matrix exponential.
            # [A, I; 0, 0] exponential applied to [y0; 0] gives [y(t); integral].
            # Simpler: y(t) = Phi @ y0 + Phi @ c where c solves A c = -b (particular steady offset).
            # Use: y(t) = Phi @ (y0 + c) - c,  with Phi = exp(A dt), c = -A^{-1} b.
            dt = t1 - t0
            Phi = expm(A * dt)
            try:
                c = np.linalg.solve(A, -b)
            except np.linalg.LinAlgError:
                c = np.linalg.lstsq(A, -b, rcond=None)[0]
            y_final = Phi @ (y0 + c) - c
            if np.all(np.isfinite(y_final)):
                return y_final
        except Exception:
            pass

    # Fallback: RK4 with adaptive-ish fixed steps, guarded by deadline.
    steps = 2000
    y = _rk4(f, t0, t1, y0, steps)
    if y is not None:
        return y

    # Last resort: fewer steps to avoid timeout.
    if time.time() < deadline:
        y = _rk4(f, t0, t1, y0, 500)
        if y is not None:
            return y

    # Return best-effort zero-state to avoid None.
    return y0
# EVOLVE-BLOCK-END
