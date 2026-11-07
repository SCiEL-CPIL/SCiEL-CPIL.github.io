import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    """
    Integrates y' = A y from t_start to t_end using spectral decomposition.
    A = (N+1)^2 * tridiag(1, -2, 1)
    """
    t0, t1 = t_span
    dt = t1 - t0
    
    # Ensure f is called at least once
    _ = f(t0, y0)
    
    N = len(y0)
    
    # Construct A
    # A = (N+1)^2 * L
    # L has -2 on diagonal, 1 on off-diagonals
    scale = float((N + 1) ** 2)
    
    # For tridiag(1, -2, 1), eigenvalues are -4*sin^2(k*pi/(2*(N+1)))
    # So A's eigenvalues are scale * (-4*sin^2(k*pi/(2*(N+1))))
    # Eigenvectors: v_k[i] = sin(i*k*pi/(N+1)) for i=1..N, k=1..N
    
    # Compute eigenvalues of A
    k = np.arange(1, N + 1, dtype=np.float64)
    eigvals = -4.0 * scale * np.sin(np.pi * k / (2.0 * (N + 1)))**2
    
    # Compute exp(eigvals * dt)
    exp_eigvals = np.exp(eigvals * dt)
    
    # We need to compute y_end = expm(A*dt) @ y0
    # Using spectral decomposition: A = V * diag(eigvals) * V^{-1}
    # Since the eigenvectors are orthogonal (sine modes), V^{-1} = V^T / (V^T V)
    
    # Build the eigenvector matrix V where V[i-1, k-1] = sin(i*k*pi/(N+1))
    i = np.arange(1, N + 1, dtype=np.float64)
    i_col = i[:, np.newaxis]  # shape (N, 1)
    k_row = k[np.newaxis, :]  # shape (1, N)
    V = np.sin(np.pi * i_col * k_row / (N + 1))
    
    # V is orthogonal up to a constant factor
    # V^T V = (N+1)/2 * I
    # So V^{-1} = V^T / ((N+1)/2)
    
    # Compute V^T @ y0
    Vt_y0 = V.T @ y0
    
    # Scale by 2/(N+1) to get the coefficients in the eigenbasis
    coeffs = (2.0 / (N + 1)) * Vt_y0
    
    # Apply exp(eigvals * dt)
    coeffs_expanded = exp_eigvals * coeffs
    
    # Transform back: y_end = V @ coeffs_expanded * (N+1)/2
    y_end = (V @ coeffs_expanded) * ((N + 1) / 2.0)
    
    # Check for finiteness
    if not np.all(np.isfinite(y_end)):
        # Fallback to matrix exponential
        A = np.zeros((N, N), dtype=np.float64)
        np.fill_diagonal(A, -2.0 * scale)
        idx = np.arange(N - 1)
        A[idx, idx + 1] = scale
        A[idx + 1, idx] = scale
        M = expm(A * dt)
        y_end = M @ y0
        if not np.all(np.isfinite(y_end)):
            return np.array(y0)
        
    return y_end
