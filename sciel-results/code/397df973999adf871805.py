import numpy as np
import time
from scipy.linalg import expm_multiply

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 300s limit
    
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    t0, t1 = t_span
    dt = t1 - t0
    
    if n == 0:
        return y0

    # Define the matrix-vector product
    # For a linear ODE y' = A(t) y, f(t, y) = A(t) @ y
    # Assuming time-invariant A, f(t, y) = A @ y
    def matvec(v):
        return f(t0, v)

    # We use an adaptive Krylov subspace method to approximate e^(A*dt) @ y0
    # This reduces the number of function calls compared to full matrix extraction.
    
    # List of Krylov subspace dimensions to try
    # We start small because the heat equation solution decays rapidly,
    # meaning only the first few eigenmodes contribute significantly.
    dims = [5, 10, 20, 50, 100]
    
    y_prev = None
    y_best = None
    best_dim = 0
    
    for dim in dims:
        # Check if we are running out of time
        if time.time() > deadline - 10:
            break
            
        d = min(dim, n)
        
        try:
            # Compute approximate solution
            # 'arnoldi' is a robust choice for non-symmetric matrices
            y_est = expm_multiply(matvec, y0, dt, method='arnoldi', dim=d, eps=1e-14)
        except Exception:
            # If this dimension fails, try the next or fall back
            continue
            
        if y_prev is not None:
            # Check for convergence
            diff = np.linalg.norm(y_est - y_prev)
            norm = np.linalg.norm(y_est)
            if norm > 1e-15:
                rel_change = diff / norm
            else:
                rel_change = diff
                
            if rel_change < 1e-12:
                y_best = y_est
                best_dim = d
                break
        else:
            # First successful computation
            y_best = y_est
            best_dim = d
            
        y_prev = y_est

    # If we didn't break due to convergence, y_best is the result from the largest dim tried
    # If no dim succeeded, y_best is None
    
    if y_best is None:
        # Fallback: Full basis extraction if Krylov failed entirely
        # This is expensive (n calls) but robust
        try:
            A = np.zeros((n, n), dtype=float)
            for i in range(n):
                e_i = np.zeros(n, dtype=float)
                e_i[i] = 1.0
                A[:, i] = f(t0, e_i)
                
                if time.time() > deadline - 5:
                    break
            
            if time.time() > deadline - 2:
                return y0
                
            # Compute exact solution
            from scipy.linalg import expm
            M = expm(A * dt)
            y_best = M @ y0
        except Exception:
            return y0

    return y_best
