import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Legendre polynomial expansion.
    
    The input x is in [-3, 3]. We scale to u in [-1, 1] for Legendre polynomials.
    We use 10 parameters for Legendre polynomials L_0 through L_9.
    
    Legendre polynomials are orthogonal with respect to the uniform distribution
    on [-1, 1], which matches the uniform sampling of x in [-3, 3] after scaling.
    This should provide better conditioning and faster convergence for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale input to [-1, 1]
    u = x / 3.0
    
    n_basis = 10
    n_points = x.size
    
    # Build Legendre polynomial basis matrix using three-term recurrence
    # L_0(u) = 1
    # L_1(u) = u
    # L_{n+1}(u) = ((2n+1)*u*L_n(u) - n*L_{n-1}(u)) / (n+1)
    
    L = np.zeros((n_points, n_basis), dtype=np.float64)
    L[:, 0] = 1.0
    if n_basis > 1:
        L[:, 1] = u
    
    for k in range(2, n_basis):
        # L_k = ((2*(k-1)+1)*u*L_{k-1} - (k-1)*L_{k-2}) / k
        #     = ((2k-1)*u*L_{k-1} - (k-1)*L_{k-2}) / k
        L[:, k] = ((2 * k - 1) * u * L[:, k - 1] - (k - 1) * L[:, k - 2]) / k
    
    # Prediction: sum of params[i] * L_i(u)
    pred = L @ params
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
