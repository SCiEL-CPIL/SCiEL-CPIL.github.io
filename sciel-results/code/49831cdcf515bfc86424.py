"""Symbolic regression with hybrid Fourier + RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid basis:
    - 4 fixed-frequency sine/cosine terms (Fourier-like)
    - 6 fixed-center Gaussian RBF terms
    
    10 learnable weights total. The problem is linear in params, so BFGS
    finds the global optimum for any fixed basis configuration.
    
    Parameters (length 10):
    w0: weight for sin(1.0 * x)
    w1: weight for cos(1.0 * x)
    w2: weight for sin(2.0 * x)
    w3: weight for cos(2.0 * x)
    w4-w9: weights for 6 Gaussian RBFs at fixed centers in [-2, 2]
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Fourier part: 4 basis functions (sin/cos at frequencies 1 and 2)
    sin1 = np.sin(1.0 * x)
    cos1 = np.cos(1.0 * x)
    sin2 = np.sin(2.0 * x)
    cos2 = np.cos(2.0 * x)
    
    y = params[0] * sin1 + params[1] * cos1 + params[2] * sin2 + params[3] * cos2
    
    # RBF part: 6 fixed centers evenly spaced in [-2, 2]
    n_rbf = 6
    centers = np.linspace(-1.6, 1.6, n_rbf)  # slightly narrower to avoid edge effects
    sigma = 0.7  # moderate width for good overlap
    
    x_expanded = x[:, np.newaxis]  # (n_points, 1)
    c_expanded = centers[np.newaxis, :]  # (1, n_rbf)
    
    diff = x_expanded - c_expanded  # (n_points, n_rbf)
    rbf_basis = np.exp(-diff ** 2 / (2.0 * sigma ** 2))  # (n_points, n_rbf)
    
    # Linear combination of RBFs
    y += rbf_basis @ params[4:10]
    
    return y
# EVOLVE-BLOCK-END
