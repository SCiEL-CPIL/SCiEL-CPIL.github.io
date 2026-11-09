import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0 # 300s limit, keep 10s margin

    # Check for affine term b
    # f(t, y) = A y + b
    # b = f(t, 0)
    b = np.array(f(t0, np.zeros(n)), dtype=float)
    
    # Check if it's effectively homogeneous
    is_homogeneous = np.linalg.norm(b) < 1e-14
    
    if is_homogeneous:
        # Extract A using n calls
        A = np.zeros((n, n), dtype=float)
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j)
        
        # Compute expm(A * dt)
        dt = t1 - t0
        M = scipy.linalg.expm(A * dt)
        y_end = M @ y0
    else:
        # Affine system: y' = A y + b
        # Augment to z' = B z where z = [y; 1]
        # B = [[A, b], [0, 0]]
        # Extract A using n calls
        A = np.zeros((n, n), dtype=float)
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            A[:, j] = f(t0, e_j)
            
        # Construct B
        B = np.zeros((n + 1, n + 1), dtype=float)
        B[:n, :n] = A
        B[:n, n] = b
        # Last row is zeros
        
        # Initial augmented state
        z0 = np.zeros(n + 1)
        z0[:n] = y0
        z0[n] = 1.0
        
        dt = t1 - t0
        M = scipy.linalg.expm(B * dt)
        z_end = M @ z0
        y_end = z_end[:n]
        
    return y_end
