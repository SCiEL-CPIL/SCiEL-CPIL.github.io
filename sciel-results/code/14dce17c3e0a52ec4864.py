"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Softplus basis expansion with fixed centers and slopes.
    
    We use 10 basis functions of the form:
        w_i * softplus(s_i * (x - c_i))
    
    where softplus(z) = log(1 + exp(z)) (numerically stable version).
    
    The centers c_i are fixed to a symmetric grid in [-2, 2],
    and slopes s_i are fixed. Only weights w_i (params[0:10]) are free.
    
    Since the model is linear in the weights, BFGS will converge to
    the global minimum of the least-squares problem.
    """
    # Fixed centers: symmetric grid in [-2, 2]
    # 10 centers spread across the domain
    c = np.array([-2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    # We need exactly 10 basis functions for 10 params
    # Let's use 10 centers
    c = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 0.0])
    
    # Fixed slopes: use a range of slopes to capture different features
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    
    # Compute softplus basis: softplus(z) = log(1 + exp(z))
    # Numerically stable: for large z, softplus(z) ≈ z; for very negative z, ≈ 0
    def softplus(z):
        # Use the stable formulation: max(z, 0) + log(1 + exp(-|z|))
        return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
    
    # Build basis matrix: shape (10, len(x))
    n = len(x)
    basis = np.empty((10, n))
    for i in range(10):
        z = s[i] * (x - c[i])
        basis[i] = softplus(z)
    
    # Prediction: sum of w_i * basis_i
    # params is length 10
    pred = basis.T @ params
    
    return pred
# EVOLVE-BLOCK-END
