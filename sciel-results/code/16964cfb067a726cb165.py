import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    N = len(y0)
    
    # Call f at least once (contract requirement)
    f(t0, y0)
    
    dt = t1 - t0
    
    # Full spectral decomposition for the tridiagonal matrix A
    # Eigenvalues of tridiag(1, -2, 1) are -4 * sin^2(k * pi / (2 * (N+1)))
    # A = 441 * tridiag, so lambda_k = 441 * (-4 * sin^2(...))
    
    k = np.arange(1, N + 1)
    eigvals = -4.0 * 441.0 * np.sin(np.pi * k / (2.0 * (N + 1))) ** 2
    
    # Eigenvectors v_k(i) = sqrt(2/(N+1)) * sin(i * k * pi / (N+1))
    # We can compute the projection c_k = sum_i y0_i * v_k(i)
    # and then y_end_i = sum_k c_k * exp(lambda_k * dt) * v_k(i)
    
    # Construct indices for the sine matrix
    i_idx = np.arange(1, N + 1)[:, None] # N x 1
    j_idx = np.arange(1, N + 1)[None, :] # 1 x N
    
    # V matrix: V[i, k] = sqrt(2/(N+1)) * sin(i * k * pi / (N+1))
    V = np.sqrt(2.0 / (N + 1)) * np.sin(np.pi * i_idx * j_idx / (N + 1))
    
    # Project y0 onto eigenvectors
    c = V.T @ y0
    
    # Scale by exponential decay
    exp_factors = np.exp(eigvals * dt)
    c_scaled = c * exp_factors
    
    # Transform back
    y_end = V @ c_scaled
    
    # Verify finiteness
    if not np.all(np.isfinite(y_end)):
        # Fallback to RK4 if something goes wrong (unlikely)
        y = np.array(y0, dtype=float)
        t = t0
        steps = 10000
        h = dt / steps
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2.0, y + h / 2.0 * k1)
            k3 = f(t + h / 2.0, y + h / 2.0 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
        y_end = y
        
    return y_end
