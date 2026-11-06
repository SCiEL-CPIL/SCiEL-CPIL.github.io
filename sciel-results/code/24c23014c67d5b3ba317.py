"""Solve 15-dim stiff linear ODE by exploiting linearity: y(t) = expm((t1-t0)*A) @ y0."""
import numpy as np
import time
from scipy.linalg import expm


def integrate(f, t_span, y0):
    deadline = time.time() + 290  # safety margin before 330s kill
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # 1. Extract b = f(t0, 0)
    y_zero = np.zeros(n, dtype=float)
    b = np.asarray(f(t0, y_zero), dtype=float)
    
    # Check if b is effectively zero to avoid noise in homogeneous systems
    if np.linalg.norm(b) < 1e-12:
        b = np.zeros(n, dtype=float)

    # 2. Extract A by evaluating f at basis vectors and subtracting b
    A = np.zeros((n, n), dtype=float)
    for j in range(n):
        if time.time() > deadline:
            break
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        col = np.asarray(f(t0, e), dtype=float)
        A[:, j] = col - b

    # 3. Compute matrix exponential
    # If b is zero, we can use the smaller n x n matrix
    if np.allclose(b, 0.0):
        M = expm(dt * A)
        y_end = M @ y0
    else:
        # Augmented system
        H = np.zeros((n + 1, n + 1), dtype=float)
        H[:n, :n] = A
        H[:n, n] = b
        # H[n, :] is 0
        
        M_aug = expm(dt * H)
        
        z0 = np.zeros(n + 1, dtype=float)
        z0[:n] = y0
        z0[n] = 1.0
        
        z_end = M_aug @ z0
        y_end = z_end[:n]
        
    return y_end
