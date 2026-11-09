"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: ensure we return within 150s wall-clock
    t_start = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Hybrid RBF + Polynomial Basis
    # RBFs centered at (0,0) and the four corners of [-1, 1]^2
    # Polynomial terms to capture global trends
    
    sigma = 0.7
    inv_2sigma2 = 1.0 / (2.0 * sigma**2)
    
    # RBFs
    rbf0 = np.exp(-(x**2 + y**2) * inv_2sigma2)
    rbf1 = np.exp(-((x - 1.0)**2 + (y - 1.0)**2) * inv_2sigma2)
    rbf2 = np.exp(-((x - 1.0)**2 + (y + 1.0)**2) * inv_2sigma2)
    rbf3 = np.exp(-((x + 1.0)**2 + (y - 1.0)**2) * inv_2sigma2)
    rbf4 = np.exp(-((x + 1.0)**2 + (y + 1.0)**2) * inv_2sigma2)
    
    # Polynomial basis
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x**2
    b4 = y**2
    
    # Stack basis functions into a design matrix (n_points x 10)
    B = np.column_stack([
        rbf0, rbf1, rbf2, rbf3, rbf4,
        b0, b1, b2, b3, b4
    ])
    
    # Matrix-vector product with safety against non-finite values
    with np.errstate(over='ignore', invalid='ignore'):
        pred = B @ params
    
    # Replace any non-finite values with 0 to keep BFGS stable
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
# EVOLVE-BLOCK-END
