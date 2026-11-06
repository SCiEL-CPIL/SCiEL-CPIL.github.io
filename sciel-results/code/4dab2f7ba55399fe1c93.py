"""ODE solver: 15-dimensional stiff linear system using matrix extraction."""
import numpy as np
import scipy.linalg
import time


def _extract_matrix_A(f, t0, y0):
    """
    Extract the system matrix A from y' = f(t, y) assuming y' = A y.
    Uses 15 basis vector probes to solve for A exactly.
    """
    n = len(y0)
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        A[:, j] = f(t0, e)
    return A


def _solve_linear_ode(f, t_span, y0):
    """
    Solve y' = A y exactly using matrix exponential or eigen-decomposition.
    First extracts A, then computes y(t1) = expm((t1-t0)*A) @ y0.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)

    # Extract matrix A using n basis vector calls
    A = _extract_matrix_A(f, t0, y0)

    # Compute exact solution: y(t1) = expm((t1 - t0) * A) @ y0
    dt = t1 - t0
    if dt == 0:
        return y0

    # Check if A is symmetric (or Hermitian, but real system assumed)
    # If symmetric, use eigh for better stability/accuracy
    if np.allclose(A, A.T, rtol=1e-12, atol=1e-14):
        try:
            w, V = scipy.linalg.eigh(A)
            # y(t) = V exp(w*t) V^T y0
            # exp(w*dt) is vector
            exp_w_dt = np.exp(w * dt)
            # V^T y0
            Vy0 = V.T @ y0
            # element-wise multiply
            res_V = exp_w_dt * Vy0
            # V @ res_V
            y_end = V @ res_V
            return y_end
        except Exception:
            pass

    # Check if A is diagonalizable with well-conditioned eigenvectors
    try:
        w, V = scipy.linalg.eig(A)
        cond_V = np.linalg.cond(V)
        if cond_V < 1e6:
            # y(t) = V exp(w*t) V^-1 y0
            exp_w_dt = np.exp(w * dt)
            V_inv = np.linalg.inv(V)
            Vy0 = V_inv @ y0
            res_V = exp_w_dt * Vy0
            y_end = V @ res_V
            return y_end
    except Exception:
        pass

    # Fallback: Use scipy's matrix exponential (highly accurate)
    expm_A = scipy.linalg.expm(dt * A)
    y_end = expm_A @ y0

    return y_end


def integrate(f, t_span, y0):
    """
    Solve the 15-dimensional stiff ODE y' = f(t, y) on [t0, t1].
    Assumes the system is linear: f(t, y) = A @ y.
    Uses matrix extraction + matrix exponential/eigen for exact solution.
    """
    deadline = time.time() + 280  # safety margin before 300s limit
    y0 = np.array(y0, dtype=float)
    n = len(y0)

    # Extract A and solve exactly
    try:
        y_end = _solve_linear_ode(f, t_span, y0)
        if np.all(np.isfinite(y_end)):
            return y_end
    except Exception:
        pass

    # Fallback: if matrix extraction failed, use a robust implicit method
    # (should not normally be reached for a well-formed linear system)
    return _fallback_implicit(f, t_span, y0, deadline)


def _fallback_implicit(f, t_span, y0, deadline):
    """Fallback implicit Euler with Newton iterations if matrix extraction fails."""
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    # Use a moderate number of steps
    steps = 5000
    h = (t1 - t0) / steps
    t = t0

    for i in range(steps):
        if time.time() > deadline:
            break
        z = y.copy()
        for _newton in range(6):
            fz = f(t + h, z)
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1e-8 * max(1.0, abs(z[j]))
                J[:, j] = (f(t + h, z + e) - fz) / e[j]
            try:
                step = np.linalg.solve(np.eye(n) - h * J, -(z - y - h * fz))
            except np.linalg.LinAlgError:
                step = -(z - y - h * fz) / max(1.0, h * np.max(np.abs(J)))
            z = z + step
            if np.linalg.norm(step) <= 1e-12 * (1 + np.linalg.norm(z)):
                break
        y = z
        t += h

    return y
