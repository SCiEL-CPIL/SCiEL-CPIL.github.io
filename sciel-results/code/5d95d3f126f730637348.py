# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm


def _estimate_A(t, f, n):
    """Estimate the coefficient matrix A(t) for the linear ODE y' = A(t) y.
    Uses the fact that f(t, y) = A(t) @ y, so A[:, j] = f(t, e_j)."""
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        A[:, j] = f(t, e)
    return A


def _check_time_invariant(A1, A2, tol=1e-12):
    """Check if two matrices are approximately equal."""
    return np.linalg.norm(A1 - A2) <= tol * max(1.0, np.linalg.norm(A1))


def _solve_linear_ode(f, t_span, y0):
    """Solve y' = A(t) y using matrix exponential."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    deadline = time.time() + 280
    
    # First, check if A is time-invariant by sampling at two points
    A_start = _estimate_A(t0, f, n)
    A_mid = _estimate_A((t0 + t1) / 2, f, n)
    
    if _check_time_invariant(A_start, A_mid):
        # Time-invariant: y_end = exp(A * (t1 - t0)) @ y0
        dt = t1 - t0
        Phi = expm(A_start * dt)
        return Phi @ y0
    
    # Time-varying: use piecewise-constant approximation
    best_result = None
    prev_result = None
    
    for n_segments in [8, 16, 32, 64, 128, 256, 512, 1024]:
        if time.time() > deadline:
            break
        
        h = (t1 - t0) / n_segments
        Phi = np.eye(n, dtype=float)
        
        for i in range(n_segments):
            t_i = t0 + i * h
            A_i = _estimate_A(t_i, f, n)
            M_i = expm(A_i * h)
            Phi = M_i @ Phi
        
        y_end = Phi @ y0
        
        if prev_result is not None:
            diff = np.linalg.norm(y_end - prev_result) / max(1.0, np.linalg.norm(y_end))
            if diff < 1e-13:
                return y_end
        
        prev_result = y_end
        best_result = y_end
    
    return best_result


def integrate(f, t_span, y0):
    """
    Integrate the 15-dimensional stiff linear ODE y' = f(t, y) on [t0, t1].
    Exploits linearity: f(t, y) = A(t) @ y, so use matrix exponential.
    """
    return _solve_linear_ode(f, t_span, y0)
# EVOLVE-BLOCK-END
