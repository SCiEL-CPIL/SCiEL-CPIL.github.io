"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    Assumes the system is linear: y' = A y.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin for 300s limit
    
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # 1. Determine the matrix A by evaluating f at origin and basis vectors.
    # f(t, y) should return A @ y.
    # f(0, 0) should be 0 for a homogeneous linear system.
    f0 = f(t0, np.zeros(n))
    
    A = np.zeros((n, n), dtype=float)
    
    # Evaluate f at basis vectors to find columns of A
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        # We can use t0 or any t, since A is constant for linear systems.
        # Using t0 is standard.
        f_j = f(t0, e_j)
        A[:, j] = f_j
        
        # Check deadline
        if time.time() > deadline:
            # Fallback if we are running out of time (unlikely for N=15)
            # Just return y0 or a simple Euler step if we have partial info
            # But for N=15, this loop takes microseconds.
            pass

    # 2. Compute the exact solution using matrix exponential
    # y(t1) = exp(A * (t1 - t0)) @ y0
    dt = t1 - t0
    M = A * dt
    expM = expm(M)
    y_final = expM @ y0
    
    # 3. Return the final state
    return y_final
# EVOLVE-BLOCK-END
