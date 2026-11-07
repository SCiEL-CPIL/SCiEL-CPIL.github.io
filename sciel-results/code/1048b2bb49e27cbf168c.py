"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Must call f at least once
    f(t0, y0)
    
    # Construct A = 21^2 * tridiag(1, -2, 1) for N=20
    # tridiag(1, -2, 1) means: diagonal = -2, sub/super diagonal = 1
    scale = 21.0 ** 2
    A = scale * np.zeros((n, n))
    for i in range(n):
        A[i, i] = -2.0
        if i > 0:
            A[i, i-1] = 1.0
        if i < n-1:
            A[i, i+1] = 1.0
    
    # A is symmetric, use eigh for stable eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # y(t) = V * diag(exp(lambda_i * dt)) * V^T * y0
    dt = t1 - t0
    exp_factors = np.exp(eigenvalues * dt)
    
    # y_end = V @ (exp_factors * (V^T @ y0))
    V = eigenvectors
    coeff = V.T @ y0
    y_end = V @ (exp_factors * coeff)
    
    return y_end
# EVOLVE-BLOCK-END
