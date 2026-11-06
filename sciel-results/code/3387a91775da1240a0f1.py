import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0
    
    # Step 1: Reconstruct the tridiagonal operator A by probing f
    # f(t, y) = A @ y (autonomous ODE for convection-diffusion)
    # Call f(t0, y0) to get A@y0
    # Call f(t0, e_j) for each basis vector to get columns of A
    
    # First, get A@y0 as a sanity check / for potential use
    Ay0 = f(t0, y0)
    
    # Reconstruct A column by column
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        A[:, j] = f(t0, e)
    
    # Step 2: Extract tridiagonal structure
    # For a convection-diffusion operator, A should be tridiagonal
    # Extract the three diagonals
    diag = np.diag(A)
    upper = np.diag(A, 1)  # superdiagonal
    lower = np.diag(A, -1)  # subdiagonal
    
    # Step 3: Compute eigenvalues of the tridiagonal Toeplitz matrix
    # For a tridiagonal Toeplitz matrix with diagonal d, superdiag c1, subdiag c2:
    # If c1 == c2 (symmetric case): lambda_k = d + 2*c*cos(k*pi/(n+1))
    # For general tridiagonal Toeplitz: we can use scipy or compute directly
    # 
    # Actually, for a general tridiagonal Toeplitz matrix (not necessarily symmetric),
    # the eigenvalues can still be computed analytically.
    # 
    # Let's check if it's Toeplitz (constant diagonals)
    d = np.mean(diag)
    c1 = np.mean(upper) if len(upper) > 0 else 0.0
    c2 = np.mean(lower) if len(lower) > 0 else 0.0
    
    # Check if it's truly Toeplitz (all diagonal entries are the same)
    is_toeplitz = (np.allclose(diag, d, atol=1e-10) and 
                   (len(upper) == 0 or np.allclose(upper, c1, atol=1e-10)) and
                   (len(lower) == 0 or np.allclose(lower, c2, atol=1e-10)))
    
    if is_toeplitz:
        # Compute eigenvalues analytically for tridiagonal Toeplitz
        # For matrix with diagonal d, superdiagonal c1, subdiagonal c2:
        # lambda_k = d + 2*sqrt(c1*c2)*cos(k*pi/(n+1)) for k=1,...,n
        # This is valid when c1*c2 >= 0
        # 
        # Actually, the standard result is for the symmetric case c1=c2.
        # For the general case, we use:
        # lambda_k = d + 2*sqrt(c1*c2)*cos(k*pi/(n+1))
        
        if c1 * c2 >= 0:
            sqrt_c1c2 = np.sqrt(c1 * c2)
            k = np.arange(1, n + 1)
            eigenvalues = d + 2 * sqrt_c1c2 * np.cos(k * np.pi / (n + 1))
        else:
            # Complex eigenvalues case - use scipy
            from scipy.linalg import eig
            eigenvalues, eigenvectors = eig(A)
            # Compute matrix exponential directly
            from scipy.linalg import expm
            M = expm(A * dt)
            return M @ y0
    else:
        # Not Toeplitz - use scipy
        from scipy.linalg import expm
        M = expm(A * dt)
        return M @ y0
    
    # Step 4: Compute eigenvectors analytically
    # For tridiagonal Toeplitz, eigenvectors are:
    # v_k(j) = sin(j * k * pi / (n+1)) for the symmetric case
    # For the general case with c1 != c2, we need to be more careful
    
    if c1 == c2 or np.isclose(c1, c2, atol=1e-14):
        # Symmetric case: eigenvectors are sine functions
        k = np.arange(1, n + 1)
        j = np.arange(1, n + 1)
        V = np.sin(np.outer(j, k) * np.pi / (n + 1))
        # Normalize
        norms = np.linalg.norm(V, axis=0)
        V = V / norms
    else:
        # General case: use scipy to get eigenvectors
        from scipy.linalg import eig
        eigenvalues, V = eig(A)
        # We need V^{-1} @ y0, then multiply by exp(lambda_i * dt), then V @ result
        # Actually, y(t) = V @ diag(exp(lambda_i * dt)) @ V^{-1} @ y0
        try:
            V_inv = np.linalg.inv(V)
            Vy0 = V_inv @ y0
            scaled = Vy0 * np.exp(eigenvalues * dt)
            return V @ scaled
        except np.linalg.LinAlgError:
            from scipy.linalg import expm
            M = expm(A * dt)
            return M @ y0
    
    # Step 5: Compute y(t1) = V @ diag(exp(lambda_i * dt)) @ V^{-1} @ y0
    # Using the analytical eigendecomposition
    try:
        V_inv = np.linalg.inv(V)
        Vy0 = V_inv @ y0
        scaled = Vy0 * np.exp(eigenvalues * dt)
        result = V @ scaled
        return result
    except np.linalg.LinAlgError:
        # Fallback to scipy
        from scipy.linalg import expm
        M = expm(A * dt)
        return M @ y0
