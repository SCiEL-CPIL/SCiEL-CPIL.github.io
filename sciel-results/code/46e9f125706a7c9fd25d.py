"""Rewritten program using a distinct approach: scaled Vandermonde with SVD/Tikhonov."""
# EVOLVE-BLOCK-START
import numpy as np


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-five polynomial to the subgrid tendency as a function of x.
    
    Uses a column-scaled Vandermonde matrix to improve numerical conditioning.
    Solves a Tikhonov-regularized least squares problem in the scaled basis,
    then maps back to the raw monomial coefficients.
    """
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    # Filter out any non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Build the raw Vandermonde matrix
    # V[i, j] = x[i]^j
    V_raw = np.vander(x, N=6, increasing=True)
    
    # Scale each column to unit L2 norm to improve conditioning
    # s_j = ||V_raw[:, j]||_2
    s = np.linalg.norm(V_raw, axis=0)
    
    # Handle zero columns (should not happen for j=0, but good for safety)
    s[s < 1e-12] = 1.0
    
    V_scaled = V_raw / s
    
    # Tikhonov regularization
    # G = V_scaled^T V_scaled
    G = V_scaled.T @ V_scaled
    b = V_scaled.T @ y
    
    # Regularization parameter
    # Use a small relative regularization
    # trace_G is sum of eigenvalues
    trace_G = np.trace(G)
    if trace_G > 0:
        lambda_reg = 1e-5 * trace_G / 6.0
    else:
        lambda_reg = 1e-5
    
    A = G + lambda_reg * np.eye(6)
    
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Map back to raw coefficients: c_j = w_j / s_j
    c = w / s
    
    # Final sanity check
    if not np.all(np.isfinite(c)):
        c = np.zeros(6)
    
    return c
# EVOLVE-BLOCK-END
