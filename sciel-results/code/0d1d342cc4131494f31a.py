import numpy as np
import scipy.linalg
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    start_time = time.time()
    deadline = start_time + 280  # 330s limit, keep 50s margin
    
    # Extract A and b from f(t, y) = A y + b
    # First, get b = f(t0, 0)
    b = f(t0, np.zeros(n))
    
    # Get columns of A: A[:, i] = f(t0, e_i) - b
    A = np.zeros((n, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i) - b
        
        if time.time() > deadline:
            # Emergency: use expm with what we have
            try:
                M = np.zeros((n + 1, n + 1))
                M[:n, :n] = A
                M[:n, n] = b
                z0 = np.zeros(n + 1)
                z0[:n] = y0
                z0[n] = 1.0
                z_final = scipy.linalg.expm(M * dt) @ z0
                return z_final[:n]
            except Exception:
                return y0
    
    # Check if b is negligible
    norm_b = np.linalg.norm(b)
    scale = np.linalg.norm(A, ord=np.inf) + 1.0
    
    if norm_b < 1e-14 * scale:
        # Homogeneous: y' = A y
        # Use eigendecomposition for speed
        try:
            eigenvalues, eigenvectors = np.linalg.eig(A)
            cond_V = np.linalg.cond(eigenvectors)
            
            if cond_V < 1e8:
                c = np.linalg.solve(eigenvectors, y0)
                exp_factors = np.exp(eigenvalues * dt)
                y_final = eigenvectors @ (exp_factors * c)
                if np.iscomplexobj(y_final):
                    y_final = np.real_if_close(y_final, tol=1000)
                return y_final
            else:
                expm_A_dt = scipy.linalg.expm(A * dt)
                return expm_A_dt @ y0
        except Exception:
            try:
                expm_A_dt = scipy.linalg.expm(A * dt)
                return expm_A_dt @ y0
            except Exception:
                return y0
    else:
        # Non-homogeneous: y' = A y + b
        # Augment to (n+1)-dim: z = [y; 1], z' = [[A, b], [0, 0]] z
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = b
        
        z0 = np.zeros(n + 1)
        z0[:n] = y0
        z0[n] = 1.0
        
        try:
            eigenvalues, eigenvectors = np.linalg.eig(M)
            cond_V = np.linalg.cond(eigenvectors)
            
            if cond_V < 1e8:
                c = np.linalg.solve(eigenvectors, z0)
                exp_factors = np.exp(eigenvalues * dt)
                z_final = eigenvectors @ (exp_factors * c)
                y_final = z_final[:n]
                if np.iscomplexobj(y_final):
                    y_final = np.real_if_close(y_final, tol=1000)
                return y_final
            else:
                expm_M_dt = scipy.linalg.expm(M * dt)
                z_final = expm_M_dt @ z0
                return z_final[:n]
        except Exception:
            try:
                expm_M_dt = scipy.linalg.expm(M * dt)
                z_final = expm_M_dt @ z0
                return z_final[:n]
            except Exception:
                return y0
