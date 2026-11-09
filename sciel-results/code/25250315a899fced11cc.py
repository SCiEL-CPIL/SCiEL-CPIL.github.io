"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _extract_matrix(f, t0, y0):
    """
    Extract the system matrix A for a linear ODE y' = A y using n calls.
    
    Strategy: Use y0 as the first test vector and n-1 standard unit vectors for the
    remaining columns. This gives us n linearly independent vectors (assuming y0 is
    not in the span of the first n-1 unit vectors, which is generically true).
    
    Let Y = [y0, e_0, e_1, ..., e_{n-2}] (n x n matrix)
    Let F = [f(t0, y0), f(t0, e_0), ..., f(t0, e_{n-2})] (n x n matrix)
    Then F = A @ Y, so A = F @ Y^{-1}.
    
    If Y is singular (y0 in span of e_0..e_{n-2}), fall back to using all n unit vectors
    (still n calls total).
    """
    n = len(y0)
    
    # Build Y matrix: first column is y0, remaining columns are e_0, ..., e_{n-2}
    Y = np.zeros((n, n))
    Y[:, 0] = y0
    for j in range(n - 1):
        Y[j, j + 1] = 1.0
    
    # Check if Y is well-conditioned
    try:
        det_Y = np.linalg.det(Y)
        if abs(det_Y) < 1e-10:
            # Fallback: use all n unit vectors
            A = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = 1.0
                A[:, j] = f(t0, e)
            return A
    except np.linalg.LinAlgError:
        # Fallback: use all n unit vectors
        A = np.empty((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            A[:, j] = f(t0, e)
        return A
    
    # Make n calls: f(t0, y0) and f(t0, e_j) for j = 0, ..., n-2
    F = np.empty((n, n))
    F[:, 0] = f(t0, y0)
    for j in range(n - 1):
        e = np.zeros(n)
        e[j] = 1.0
        F[:, j + 1] = f(t0, e)
    
    # A = F @ Y^{-1}
    Y_inv = np.linalg.inv(Y)
    A = F @ Y_inv
    
    return A


def _solve_linear_ode_explicit(A, t0, t1, y0):
    """
    Solve y' = A y using matrix exponential via scipy.
    """
    from scipy.linalg import expm
    dt = t1 - t0
    M = expm(A * dt)
    return M @ y0


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract A using n calls (leveraging y0 as first basis vector)
    A = _extract_matrix(f, t0, y0)
    
    # Solve using matrix exponential
    y_end = _solve_linear_ode_explicit(A, t0, t1, y0)
    
    return y_end
# EVOLVE-BLOCK-END
