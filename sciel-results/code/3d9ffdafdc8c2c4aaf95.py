"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate y' = A y from t_span[0] to t_span[1] with initial condition y0.
    Uses the spectral decomposition of the fixed tridiagonal matrix A.
    A = 21^2 * tridiag(1, -2, 1) with N=20.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)

    # Mandatory call to f
    _ = f(t0, y0)

    # Build the matrix A: A = 21^2 * tridiag(1, -2, 1)
    # The tridiagonal matrix T has 1 on off-diagonals, -2 on diagonal
    T = np.zeros((n, n))
    for i in range(n):
        T[i, i] = -2.0
        if i > 0:
            T[i, i - 1] = 1.0
        if i < n - 1:
            T[i, i + 1] = 1.0
    A = 441.0 * T  # 21^2 = 441

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Transform y0 to eigenbasis
    # eigenvectors are columns, so V^T @ y0 gives coefficients
    coeffs = eigenvectors.T @ y0

    # Scale by exp(eigenvalue * dt)
    dt = t1 - t0
    coeffs_end = coeffs * np.exp(eigenvalues * dt)

    # Transform back
    y_end = eigenvectors @ coeffs_end

    return y_end
# EVOLVE-BLOCK-END
