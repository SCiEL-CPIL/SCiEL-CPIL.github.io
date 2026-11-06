"""ODE solver: 15-dimensional stiff linear system using matrix extraction and matrix exponential."""
import numpy as np
import scipy.linalg
import time


def _extract_matrix_A(f, t0, y0):
    """
    Extract the system matrix A from y' = f(t, y) assuming y' = A y (autonomous linear system).
    Uses 15 random probe vectors to solve for A.
    """
    n = len(y0)
    rng = np.random.default_rng(12345)
    
    # Generate 15 random probe vectors
    probes = rng.standard_normal((n, n))
    
    # Evaluate f at each probe vector
    F = np.empty((n, n))
    for j in range(n):
        F[:, j] = f(t0, probes[:, j])
    
    # A * probes = F, so A = F * inv(probes)
    A = F @ np.linalg.inv(probes)
    
    # Verify linearity: check that f(t, 0) should be 0 for a linear system
    # We won't call f(0) to save a call, but if the system is linear, f(t,0)=0
    return A


def _solve_linear_ode(A, y0, t_span):
    """
    Solve y' = A y on [t0, t1] using the matrix exponential.
    y(t1) = expm(A * (t1 - t0)) @ y0
    """
    t0, t1 = t_span
    dt = t1 - t0
    M = scipy.linalg.expm(A * dt)
    return M @ y0


def integrate(f, t_span, y0):
    """
    Integrate the 15-dimensional stiff linear ODE y' = f(t, y) = A y on [t0, t1].
    Extracts A using 15 random probes, then uses the matrix exponential for the exact solution.
    """
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 330s kill
    
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    t0, t1 = t_span
    
    # Extract the system matrix A using 15 random probes
    A = _extract_matrix_A(f, t0, y0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return y0 if we're running out of time (shouldn't happen)
        return y0
    
    # Solve using matrix exponential
    y_end = _solve_linear_ode(A, y0, t_span)
    
    # Verify result is finite
    if not np.all(np.isfinite(y_end)):
        # Fallback to a simple explicit method if matrix exponential fails
        h = (t1 - t0) / 1000
        t, y = t0, y0.copy()
        for _ in range(1000):
            k = f(t, y)
            y = y + h * k
            t += h
            if not np.all(np.isfinite(y)):
                break
        y_end = y
    
    return y_end
