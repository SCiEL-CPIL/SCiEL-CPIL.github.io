import numpy as np
import scipy
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 285  # safety margin before 300s kill
    
    t0, t1 = t_span
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    # Extract A and b from the linear system y' = A*y + b
    # This requires n+1 calls to f
    zero_vec = np.zeros(n, dtype=float)
    b_val = np.array(f(t0, zero_vec), dtype=float)
    
    A_val = np.zeros((n, n), dtype=float)
    for i in range(n):
        e_i = np.zeros(n, dtype=float)
        e_i[i] = 1.0
        col_i = np.array(f(t0, e_i), dtype=float) - b_val
        A_val[:, i] = col_i
    
    h = t1 - t0
    
    # For the linear system y' = A*y + b, the exact solution is:
    # y(t) = exp(A*t)*y0 + integral_0^t exp(A*(t-s))*b ds
    # 
    # Using the augmented matrix approach:
    # Define z = [y; 1], then z' = A_aug * z where
    # A_aug = [[A, b], [0, 0]]
    # Then z(t) = exp(A_aug * t) * z0, and y(t) = z(t)[:n]
    
    A_aug = np.zeros((n + 1, n + 1), dtype=float)
    A_aug[:n, :n] = A_val
    A_aug[:n, n] = b_val
    
    z0 = np.zeros(n + 1, dtype=float)
    z0[:n] = y0
    z0[n] = 1.0
    
    # Attempt eigen-decomposition based exponential for potentially higher precision
    # Fall back to scipy.linalg.expm if unstable or fails
    try:
        w, V = np.linalg.eig(h * A_aug)
        V_inv = np.linalg.inv(V)
        M = V @ np.diag(np.exp(w)) @ V_inv
        z_end = M @ z0
        y_end = z_end[:n]
        
        if not np.all(np.isfinite(y_end)):
            raise ValueError("Non-finite result")
            
        return y_end
    except Exception:
        M = scipy.linalg.expm(h * A_aug)
        z_end = M @ z0
        return z_end[:n]
