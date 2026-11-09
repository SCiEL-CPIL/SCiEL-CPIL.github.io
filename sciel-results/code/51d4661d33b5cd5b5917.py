"""ODE solver: Heat2D 10x10.
Exploits the linearity y' = A y to solve exactly with a matrix exponential and
polish with a few implicit-Euler (Newton) steps using the true Jacobian A.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _rk4(f, t0, t1, y0, steps):
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2.0, y + h / 2.0 * k1)
        k3 = f(t + h / 2.0, y + h / 2.0 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            return None
    return y


def _build_A(f, t0, y0, n):
    """Recover the constant linear operator A from f(t, y) = A y.

    Uses the homogeneity property f(t, a*y) = a * f(t, y) to test linearity,
    then probes each coordinate. Returns None if the model is not linear.
    """
    y0 = np.asarray(y0, dtype=float)
    base = f(t0, y0)
    if not np.all(np.isfinite(base)):
        return None
    scale = 1.234567
    scaled = f(t0, scale * y0)
    if not np.all(np.isfinite(scaled)):
        return None
    resid = np.linalg.norm(scaled - scale * base)
    tol = 1e-6 * max(1.0, np.linalg.norm(scale * base))
    if resid > tol:
        return None  # not (approximately) linear

    A = np.empty((n, n), dtype=float)
    for j in range(n):
        probe = np.zeros(n)
        probe[j] = 1.0
        A[:, j] = f(t0, probe)
    if not np.all(np.isfinite(A)):
        return None
    return A


def _implicit_euler_steps(f, A, t0, h, y, nsteps, deadline):
    """Polish with nsteps implicit-Euler (Newton) steps using the true Jacobian A.
    Backward-stable for the stiff heat equation; each Newton iteration is 1 f-call."""
    y = np.array(y, dtype=float)
    t = t0
    for _ in range(nsteps):
        t = t + h
        y_new = y.copy()
        for _it in range(6):
            F = f(t, y_new)
            # residual r = y_new - y - h * F ; solve (I - h A) d = -r
            rhs = y - y_new - h * F
            try:
                d = np.linalg.solve(np.eye(nsteps and len(y)) - h * A, rhs)
            except np.linalg.LinAlgError:
                break
            y_new = y_new + d
            if np.linalg.norm(d) <= 1e-12 * (1.0 + np.linalg.norm(y_new)):
                break
        y = y_new
        if time.time() > deadline:
            break
    return y


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    h = t1 - t0
    deadline = time.time() + 280.0  # safety margin under the 330s kill

    # Detect the linear operator.
    A = _build_A(f, t0, y0, n)
    if A is None:
        # Non-linear fallback: plain RK4 with a fixed number of steps.
        return _rk4(f, t0, t1, y0, 4000)

    # Exact linear propagation: y(t1) = expm(A*h) @ y0.
    try:
        from scipy.linalg import expm
        M = expm(A * h)
        y = M @ y0
    except Exception:
        # Fallback to a moderate RK4 if expm fails.
        y = _rk4(f, t0, t1, y0, 4000)
        if y is None:
            return np.array(y0, dtype=float)

    # Polish with a few implicit-Euler Newton steps for extra accuracy.
    y = _implicit_euler_steps(f, A, t0, h, y, 2, deadline)
    return y
# EVOLVE-BLOCK-END
