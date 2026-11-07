"""
STRUCTURAL CHANGE: Radial Basis Function (RBF) architecture with polynomial trend.

Combines the linear-in-params structure (from inspiration 2, which scored much
better than the nonlinear tanh approach) with a localized RBF representation
that can capture non-periodic, smooth targets more effectively than global
trigonometric or power-series bases.

Architecture (10 parameters, all linear in params for reliable BFGS convergence):
- 5 Gaussian RBF kernels centered at {-3, -1.5, 0, 1.5, 3} with fixed width
- 5 low-order polynomial terms: 1, x, x^2, x^3, x^4

The RBF terms provide localized "bumps" that can adapt to local features,
while the polynomial terms capture the global trend. Since the model is linear
in params, BFGS solves a convex problem and converges reliably from all-ones.

The RBF width is chosen to be moderate (sigma=1.0) so that neighboring centers
have some overlap, allowing smooth interpolation. On [-3,3], the RBFs decay
quickly enough to provide locality but overlap enough for continuity.
"""
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # RBF centers and fixed width
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.0  # fixed width for Gaussian kernels
    
    # Compute 5 Gaussian RBF values: shape (n, 5)
    # phi_i(x) = exp(-(x - c_i)^2 / (2 * sigma^2))
    diff = x[:, np.newaxis] - centers[np.newaxis, :]  # (n, 5)
    rbf = np.exp(-diff ** 2 / (2.0 * sigma ** 2))     # (n, 5)
    
    # Polynomial basis (5 terms)
    poly = np.column_stack([
        np.ones_like(x),   # 1
        x,                 # x
        x ** 2,            # x^2
        x ** 3,            # x^3
        x ** 4             # x^4
    ])  # (n, 5)
    
    # Stack into single basis matrix: shape (n, 10)
    B = np.column_stack([rbf, poly])  # (n, 10)
    
    # Linear combination
    y = B @ p
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
