"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    if n == 0:
        return y0
    if t0 == t1:
        return y0
        
    # Approach: Trust the task contract that the system is a constant-coefficient
    # linear system y' = A y. 
    # 1. Extract A using basis vectors (n calls).
    # 2. Compute the matrix exponential of (t1-t0)*A.
    # 3. Apply to y0.
    
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        e_i = np.zeros(n, dtype=float)
        e_i[i] = 1.0
        # f(t0, e_i) gives the i-th column of A
        A[:, i] = f(t0, e_i)
        
    dt = t1 - t0
    # Use scipy.linalg.expm for the matrix exponential
    Phi = scipy.linalg.expm(dt * A)
    
    y_end = Phi @ y0
    
    return y_end
# EVOLVE-BLOCK-END
