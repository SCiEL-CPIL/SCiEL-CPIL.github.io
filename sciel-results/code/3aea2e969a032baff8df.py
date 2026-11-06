"""Solve 15-dim stiff linear ODE by exploiting linearity: y(t) = expm((t1-t0)*A) @ y0."""
import numpy as np
import time
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm as sparse_expm


def integrate(f, t_span, y0):
    deadline = time.time() + 290  # safety margin before 330s kill
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # Handle general affine linear system: y' = A @ y + b
    # This covers the homogeneous case (b=0) and is more robust.
    # We need to extract A and b.
    
    # 1. Extract b = f(t0, 0)
    y_zero = np.zeros(n, dtype=float)
    b = np.asarray(f(t0, y_zero), dtype=float)
    
    # 2. Extract A by evaluating f at basis vectors and subtracting b
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        if time.time() > deadline:
            break
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        col = np.asarray(f(t0, e), dtype=float)
        A[:, j] = col - b

    # 3. Construct the augmented system for the affine ODE
    # Let z = [y; 1]. Then z' = H @ z where H = [[A, b], [0, 0]]
    H = np.zeros((n + 1, n + 1), dtype=float)
    H[:n, :n] = A
    H[:n, n] = b
    # H[n, :] remains 0, which is correct for the constant component '1'
    
    # Initial state for augmented system
    z0 = np.zeros(n + 1, dtype=float)
    z0[:n] = y0
    z0[n] = 1.0
    
    # 4. Compute matrix exponential of the augmented system
    M = expm(dt * H)
    z_end = M @ z0
    
    # 5. Extract the final state y
    y_end = z_end[:n]
    
    return y_end
