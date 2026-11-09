"""Symbolic regression with 1 input(s) using a hybrid Polynomial-Fourier-Exponential basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a linear combination of a hybrid basis:
    - Low-order polynomials (1, x, x^2) for global trends.
    - Fourier terms (sin/cos) for oscillatory behavior.
    - Exponential term for decay/growth.
    
    This basis is chosen to cover a wide range of common functional forms
    likely to appear in symbolic regression tasks.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    # Build the basis matrix
    # Shape: (n_points, 10)
    basis = np.zeros((n_points, 10), dtype=np.float64)
    
    # 1. Constant term
    basis[:, 0] = 1.0
    
    # 2. Linear term
    basis[:, 1] = x
    
    # 3. Quadratic term
    basis[:, 2] = x ** 2
    
    # 4-5. Fundamental frequency (2*pi*x)
    basis[:, 3] = np.sin(2.0 * np.pi * x)
    basis[:, 4] = np.cos(2.0 * np.pi * x)
    
    # 6-7. Second harmonic (4*pi*x)
    basis[:, 5] = np.sin(4.0 * np.pi * x)
    basis[:, 6] = np.cos(4.0 * np.pi * x)
    
    # 8-9. Third harmonic (8*pi*x)
    basis[:, 7] = np.sin(8.0 * np.pi * x)
    basis[:, 8] = np.cos(8.0 * np.pi * x)
    
    # 10. Exponential decay
    basis[:, 9] = np.exp(-0.5 * x)
    
    # Linear combination
    # params is shape (10,)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
