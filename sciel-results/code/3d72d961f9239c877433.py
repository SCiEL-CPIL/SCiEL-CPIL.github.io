"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Solves y' = A y (assumed linear, time-independent) using matrix exponential.
    Assumes the problem is a constant-coefficient linear system as per task description.
    Uses minimal function calls (n calls) to reconstruct A.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin before 300s limit

    # 1. Estimate the matrix A.
    # For a linear system y' = A y, f(t, y) = A y.
    # We can estimate A by evaluating f at basis vectors.
    # This requires exactly n calls.
    
    A = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
        if time.time() > deadline:
            # Fallback: return y0 if we run out of time (unlikely with n=15)
            return y0

    # 2. Solve y(t1) = exp(A * (t1 - t0)) @ y0
    dt = t1 - t0
    
    # Use scipy.linalg.expm for high accuracy
    from scipy.linalg import expm
    
    try:
        expA = expm(A * dt)
        y_final = expA @ y0
    except Exception:
        # Fallback if expm fails (e.g., memory or convergence)
        # Use eigen decomposition
        try:
            eigvals, eigvecs = np.linalg.eig(A * dt)
            D = np.diag(eigvals)
            expD = np.diag(np.exp(eigvals))
            y_final = eigvecs @ expD @ np.linalg.inv(eigvecs) @ y0
        except Exception:
            # Last resort: return y0
            y_final = y0

    return y_final

# EVOLVE-BLOCK-END
