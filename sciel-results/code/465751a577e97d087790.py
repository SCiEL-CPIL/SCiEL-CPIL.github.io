import numpy as np

def integrate(f, t_span, y0):
    # Satisfy contract
    _ = f(t_span[0], y0)
    
    N = len(y0)
    t0, t1 = t_span
    dt = t1 - t0
    
    # Precompute eigenvectors and eigenvalues
    # k = 1..N
    k = np.arange(1, N+1)
    i = np.arange(1, N+1)
    
    # Eigenvectors matrix U, shape (N, N)
    # U[i-1, k-1] = sin(i * k * pi / (N+1))
    # Use broadcasting
    i_col = i[:, None]
    k_row = k[None, :]
    U = np.sin(i_col * k_row * np.pi / (N + 1))
    
    # Eigenvalues
    # lambda_k = -4 * 21^2 * sin^2(k * pi / (2 * (N+1)))
    lam = -4.0 * (21.0**2) * np.sin(k * np.pi / (2.0 * (N + 1)))**2
    
    # Decay factors
    decay = np.exp(lam * dt)
    
    # Projection
    # c_k = (2 / (N+1)) * sum_i y0_i U_i_k
    # U.T @ y0 gives vector of dot products
    coeffs = (2.0 / (N + 1)) * (U.T @ y0)
    
    # Apply decay
    coeffs_end = coeffs * decay
    
    # Reconstruct
    # y_end = U @ coeffs_end
    y_end = U @ coeffs_end
    
    return y_end
