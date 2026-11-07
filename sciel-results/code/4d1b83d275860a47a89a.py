"""Symbolic regression with 1 input; Hybrid Chebyshev + RBF basis for flexible fitting."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining Chebyshev polynomials (global trend) and RBFs (local detail).
    
    y = sum_{k=0}^{3} p_k * T_k(2x/L - 1) + sum_{j=0}^{5} p_{4+j} * exp(-((x - c_j)^2) / (2 * sigma^2))
    
    where:
    - L = 6.0 (max domain)
    - c_j are 6 equally spaced centers in [0.1, 6.0]
    - sigma = 1.0 (fixed bandwidth)
    
    This allows the model to capture both global polynomial trends and local variations.
    """
    x = np.asarray(x, dtype=float)
    
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev evaluation
    L = 6.0
    z = 2.0 * x / L - 1.0
    z = np.clip(z, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 through T_3
    T = np.zeros((4, x.shape[0]))
    T[0, :] = 1.0
    T[1, :] = z
    for n in range(2, 4):
        T[n, :] = 2.0 * z * T[n-1, :] - T[n-2, :]
    
    # First 4 parameters: Chebyshev coefficients
    y_cheb = np.dot(params[:4], T)
    
    # RBF centers: 6 equally spaced points in [0.1, 6.0]
    centers = np.linspace(0.1, 6.0, 6)
    sigma = 1.0
    
    # Compute RBF terms: exp(-((x - c_j)^2) / (2 * sigma^2))
    # x is shape (N,), centers is shape (6,)
    # We need to compute for each center
    rbf_terms = np.zeros((6, x.shape[0]))
    for j in range(6):
        diff = x - centers[j]
        rbf_terms[j, :] = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Last 6 parameters: RBF weights
    y_rbf = np.dot(params[4:10], rbf_terms)
    
    # Sum both components
    y = y_cheb + y_rbf
    
    return y
# EVOLVE-BLOCK-END
