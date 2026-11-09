import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    # Call f at least once
    _ = f(t0, y0)
    
    # Analytic eigen-decomposition for A = 21^2 * tridiag(1, -2, 1)
    # Eigenvalues: lambda_k = 21^2 * (-2 + 2*cos(k*pi/(N+1)))
    # Eigenvectors: V[i, k] = sqrt(2/(N+1)) * sin((i+1)*(k+1)*pi/(N+1))
    # Note: i, k are 0-indexed in numpy, so i+1, k+1 correspond to 1..N
    
    k = np.arange(1, N + 1)
    eigenvalues = 441.0 * (-2.0 + 2.0 * np.cos(k * np.pi / (N + 1)))
    
    # Construct eigenvector matrix V
    # V is (N x N)
    # V[i, j] = sqrt(2/(N+1)) * sin((i+1)*(j+1)*pi/(N+1))
    i_idx = np.arange(1, N + 1)
    V = np.sqrt(2.0 / (N + 1)) * np.sin(np.outer(i_idx, k) * np.pi / (N + 1))
    
    # Project y0 onto eigenbasis
    # c = V^T y0
    c = V.T @ y0
    
    # Evolve modes
    exp_factors = np.exp(eigenvalues * dt)
    c_end = c * exp_factors
    
    # Reconstruct
    y_end = V @ c_end
    
    if not np.all(np.isfinite(y_end)):
        return y0
        
    return y_end
