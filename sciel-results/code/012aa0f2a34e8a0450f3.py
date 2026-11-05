"""Linear-operator extraction + matrix exponential for convection-diffusion 100-point IVP."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.linalg import expm, eigh, eig


def _extract_A(f, t0, y0):
    """
    Extract the system matrix A from f(t, y) = A @ y for a linear time-invariant system.
    Uses column-by-column probing with unit vectors.
    Returns A (n x n) and the number of f calls used.
    """
    n = len(y0)
    A = np.zeros((n, n), dtype=float)
    calls = 0

    # First, check if f(0, 0) is zero (confirms linearity, no constant term)
    f_zero = f(t0, np.zeros(n, dtype=float))
    calls += 1
    if np.max(np.abs(f_zero)) > 1e-10:
        # Not purely linear through origin; try to detect affine: f(t,y) = A@y + b
        # For convection-diffusion with zero BC, b should be 0.
        # If not, we'll handle it as affine.
        b = f_zero
    else:
        b = np.zeros(n, dtype=float)

    # Probe each column of A
    for j in range(n):
        e_j = np.zeros(n, dtype=float)
        e_j[j] = 1.0
        f_j = f(t0, e_j)
        calls += 1
        A[:, j] = f_j - b

    return A, b, calls


def _solve_via_expm(A, b, t0, t1, y0):
    """
    Solve y' = A@y + b for constant b using matrix exponential.
    If b is zero: y(t1) = expm(A*(t1-t0)) @ y0
    If b is nonzero (affine): y(t1) = expm(A*dt)@y0 + (expm(A*dt) - I) @ A^{-1} @ b
    """
    dt = t1 - t0
    n = len(y0)

    if np.max(np.abs(b)) < 1e-14:
        # Purely linear
        M = expm(A * dt)
        return M @ y0
    else:
        # Affine: y' = A y + b
        M = expm(A * dt)
        # Particular solution for constant b: y_p = -A^{-1} b (if A invertible)
        try:
            A_inv_b = np.linalg.solve(A, -b)
            y_p = A_inv_b
        except np.linalg.LinAlgError:
            # If A is singular, use pseudo-inverse
            A_inv_b = np.linalg.lstsq(A, -b, rcond=None)[0]
            y_p = A_inv_b

        y_h = y0 - y_p  # homogeneous part
        y_h_end = M @ y_h
        y_end = y_h_end + y_p
        return y_end


def _solve_via_eig(A, b, t0, t1, y0):
    """
    Alternative: eigen-decomposition for potentially better numerical stability
    for large matrices, or if expm is slow.
    """
    dt = t1 - t0
    n = len(y0)

    if np.max(np.abs(b)) < 1e-14:
        try:
            # Try symmetric/Hermitian first (faster, more stable)
            w, V = eigh(A)
            # y(t1) = V @ diag(exp(w*dt)) @ V^{-1} @ y0
            # For symmetric A, V is orthogonal: V^{-1} = V.T
            coeffs = V.T @ y0
            y_end = V @ (np.exp(w * dt) * coeffs)
            return y_end
        except Exception:
            pass

        # General eigen-decomposition
        w, V = eig(A)
        try:
            V_inv = np.linalg.inv(V)
            coeffs = V_inv @ y0
            y_end = V @ (np.exp(w * dt) * coeffs)
            return y_end
        except np.linalg.LinAlgError:
            return None
    else:
        # Affine case with eigen
        try:
            A_inv_b = np.linalg.solve(A, -b)
            y_p = A_inv_b
        except np.linalg.LinAlgError:
            A_inv_b = np.linalg.lstsq(A, -b, rcond=None)[0]
            y_p = A_inv_b

        y_h = y0 - y_p
        try:
            w, V = eig(A)
            V_inv = np.linalg.inv(V)
            coeffs = V_inv @ y_h
            y_h_end = V @ (np.exp(w * dt) * coeffs)
            return y_h_end + y_p
        except np.linalg.LinAlgError:
            return None


def _rk4_fallback(f, t0, t1, y0, steps=5000):
    """High-step-count RK4 as ultimate fallback."""
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
    return y


def integrate(f, t_span, y0):
    """
    Integrates the ODE y' = f(t, y) from t_span[0] to t_span[1].
    Strategy: extract linear operator A from f, then use matrix exponential
    for exact (up to expm precision) solution.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)

    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, 20s margin

    try:
        # Step 1: Extract A and b from f
        A, b, extract_calls = _extract_A(f, t0, y0)

        if time.time() > deadline:
            # Out of time; do a quick RK4
            return _rk4_fallback(f, t0, t1, y0, steps=1000)

        # Step 2: Solve via matrix exponential
        y_end = None

        # Try expm first
        try:
            y_end = _solve_via_expm(A, b, t0, t1, y0)
            if y_end is not None and np.all(np.isfinite(y_end)):
                return y_end
        except Exception:
            pass

        if time.time() > deadline:
            return _rk4_fallback(f, t0, t1, y0, steps=1000)

        # Try eigen-decomposition as alternative
        try:
            y_end = _solve_via_eig(A, b, t0, t1, y0)
            if y_end is not None and np.all(np.isfinite(y_end)):
                return y_end
        except Exception:
            pass

        # Step 3: Fallback to high-order RK if extraction or expm failed
        return _rk4_fallback(f, t0, t1, y0, steps=10000)

    except Exception:
        # Ultimate fallback
        try:
            return _rk4_fallback(f, t0, t1, y0, steps=5000)
        except Exception:
            return y0
# EVOLVE-BLOCK-END
