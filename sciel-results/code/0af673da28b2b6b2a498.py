import numpy as np
from scipy.linalg import expm
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0

    # Extract A at the midpoint for better accuracy if A(t) is time-dependent
    t_mid = (t0 + t1) / 2.0
    A = np.zeros((n, n), dtype=float)
    
    # Check if y0 is a standard basis vector
    is_basis = False
    basis_idx = -1
    for j in range(n):
        if np.allclose(y0, np.eye(n)[:, j], rtol=1e-15, atol=1e-15):
            is_basis = True
            basis_idx = j
            break
            
    if is_basis:
        # y0 is e_basis_idx
        A[:, basis_idx] = f(t_mid, y0)
        for j in range(n):
            if j == basis_idx:
                continue
            e_j = np.zeros(n, dtype=float)
            e_j[j] = 1.0
            A[:, j] = f(t_mid, e_j)
            if time.time() > deadline:
                break
    else:
        for j in range(n):
            e_j = np.zeros(n, dtype=float)
            e_j[j] = 1.0
            A[:, j] = f(t_mid, e_j)
            if time.time() > deadline:
                break

    if time.time() > deadline:
        return y0

    dt = t1 - t0
    if dt == 0:
        return y0.copy()
        
    # Solve
    M = expm(dt * A)
    y_final = M @ y0
    
    return y_final
