"""ODE solver: 15-dimensional stiff linear system using matrix extraction."""
import numpy as np
import scipy.linalg
import time


def _extract_matrix_A(f, t0, y0):
    """
    Extract the system matrix A from y' = f(t, y) assuming y' = A y.
    Uses n standard basis probes to solve for A columns.
    """
    n = len(y0)
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        probe = np.zeros(n, dtype=float)
        probe[j] = 1.0
        col = f(t0, probe)
        A[:, j] = col
    return A


def _solve_linear_ode(A, y0, t0, t1):
    """
    Solve y' = A y on [t0, t1] using matrix exponential.
    y(t1) = expm((t1-t0) * A) @ y0
    """
    dt = t1 - t0
    M = scipy.linalg.expm(dt * A)
    return M @ np.asarray(y0, dtype=float)


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # Deadline guard: kill if we've been running too long
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, keep 20s safety margin

    # Extract A using n probes (n=15 calls to f)
    A = _extract_matrix_A(f, t0, y0)

    # Check deadline
    if time.time() > deadline:
        # Fallback: return y0 (shouldn't happen for n=15)
        return y0

    # Solve using matrix exponential
    result = _solve_linear_ode(A, y0, t0, t1)

    # Check deadline
    if time.time() > deadline:
        return result

    # Verify linearity: check that f(t, 2*y0) == 2*f(t, y0) approximately
    # This is a sanity check; if it fails, the system may not be linear.
    # But the task says it IS a 15-dim stiff linear system, so we trust it.
    
    return result
