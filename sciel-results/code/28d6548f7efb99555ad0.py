import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x: (N,) array in [-2, 2]
    # params: (10,) array of coefficients
    # We use a hybrid basis: constant, linear, and Gaussian RBFs with fixed centers and widths.
    # This allows fitting local features while maintaining global trend capability.
    
    N = x.shape[0]
    
    # Basis 0: constant
    # Basis 1: linear
    # Bases 2-9: Gaussian RBFs with fixed centers and widths
    
    # Fixed centers spread across [-2, 2]
    centers = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 0.0])
    # Fixed widths: multi-scale
    widths = np.array([0.2, 0.5, 1.0, 0.2, 0.5, 1.0, 0.2, 0.5])
    
    # Build basis matrix: (N, 10)
    basis = np.zeros((N, 10))
    basis[:, 0] = 1.0
    basis[:, 1] = x
    
    for k in range(8):
        # Gaussian RBF: exp(-((x - center)^2) / (2 * width^2))
        diff = x - centers[k]
        basis[:, 2 + k] = np.exp(-diff**2 / (2.0 * widths[k]**2))
    
    return basis @ params
