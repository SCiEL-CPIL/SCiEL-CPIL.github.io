"""Solve 15-dim stiff linear ODE by exploiting linearity: y(t) = expm((t1-t0)*A) @ y0."""
import numpy as np
import time
from scipy.linalg import expm
import mpmath as mp


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
    
    # 4. Compute matrix exponential of the augmented system using mpmath for higher precision
    # Use high precision to reduce relative error
    mp.mp.dps = 50  # 50 decimal digits of precision
    
    # Convert to mpmath matrix
    H_mp = mp.matrix(n + 1, n + 1)
    for i in range(n + 1):
        for j in range(n + 1):
            H_mp[i, j] = mp.mpf(H[i, j])
    
    dt_mp = mp.mpf(dt)
    H_mp = dt_mp * H_mp
    
    # Compute matrix exponential
    M_mp = mp.expm(H_mp)
    
    # Convert z0 to mpmath vector
    z0_mp = mp.matrix(n + 1, 1)
    for i in range(n + 1):
        z0_mp[i, 0] = mp.mpf(z0[i])
    
    # Compute final state
    z_end_mp = M_mp * z0_mp
    
    # Convert back to numpy float64
    y_end = np.zeros(n, dtype=float)
    for i in range(n):
        y_end[i] = float(z_end_mp[i, 0])
    
    return y_end
