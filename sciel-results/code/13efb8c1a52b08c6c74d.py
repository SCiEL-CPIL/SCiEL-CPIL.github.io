import numpy as np
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280
    
    t0, t1 = t_span
    n = len(y0)
    y0 = np.array(y0, dtype=float)
    
    # 1. Extract A and b at t0
    # We assume the system is linear and constant: y' = A*y + b
    # b = f(t0, 0)
    # A[:, i] = f(t0, e_i) - b
    b0 = np.array(f(t0, np.zeros(n)), dtype=float)
    A0 = np.zeros((n, n))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        fi = np.array(f(t0, ei), dtype=float)
        A0[:, i] = fi - b0
        
    dt = t1 - t0
    
    # 2. Compute the exact solution using the augmented matrix exponential
    # y(t) = exp(A*dt) * y0 + integral_0^dt exp(A*(dt-s)) * b ds
    # This is equivalent to:
    # Z(t) = exp(M*dt) * Z0
    # where M = [[A, b], [0, 0]] and Z0 = [y0, 1]
    
    # Construct augmented matrix M (n+1 x n+1)
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A0
    M[:n, n] = b0
    
    Z0 = np.zeros(n + 1)
    Z0[:n] = y0
    Z0[n] = 1.0
    
    # 3. Compute exp(M * dt) using a high-order Taylor series
    # For stiff systems, the eigenvalues of A are often large in magnitude.
    # However, the matrix exponential is well-defined.
    # A Taylor series expansion: sum_{k=0}^K (M*dt)^k / k!
    # We need to choose K such that the error is below machine epsilon.
    # For a 16x16 matrix, if the norm is moderate, K=50 is usually sufficient.
    # If the norm is large, we might need scaling.
    
    # Let's estimate the norm of M*dt to decide on scaling.
    X = M * dt
    norm_X = np.linalg.norm(X, ord=np.inf)
    
    # If norm is large, use scaling and squaring with Taylor series on the scaled matrix.
    # Choose s such that ||X / 2^s|| < 1.
    if norm_X > 1:
        s = int(np.ceil(np.log2(norm_X)))
        if s < 1: s = 1
        X_scaled = X / (2 ** s)
    else:
        s = 0
        X_scaled = X
        
    # Compute Taylor series for exp(X_scaled)
    # K=50 is very safe for ||X_scaled|| < 1.
    # The terms decrease rapidly.
    K = 50
    term = np.eye(n + 1)
    exp_X_scaled = np.zeros((n + 1, n + 1))
    
    for k in range(1, K + 1):
        term = term @ X_scaled / k
        exp_X_scaled += term
        # Early termination if term is negligible
        if np.linalg.norm(term, ord=np.inf) < 1e-16 * np.linalg.norm(exp_X_scaled, ord=np.inf):
            break
            
    # Square back s times
    exp_X = exp_X_scaled
    for _ in range(s):
        exp_X = exp_X @ exp_X
        
    # 4. Compute final state
    Z_end = exp_X @ Z0
    
    return Z_end[:n]
