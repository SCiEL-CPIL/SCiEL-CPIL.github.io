"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a set of Gaussian Radial Basis Functions (RBFs)
    centered on a 3x3 grid plus a constant term.
    
    This basis allows the model to capture local variations and non-linearities
    that a global polynomial of degree 3 might miss.
    
    Basis functions:
    0: 1 (Constant)
    1-9: Gaussian RBFs centered at (xi, yi) where xi, yi in {-1, 0, 1}
    
    Gaussian: exp(-gamma * ((x - xi)^2 + (y - yi)^2))
    We use gamma = 4.0 for a balance between local flexibility and conditioning.
    """
    gamma = 4.0
    
    # Define centers
    centers_x = [-1.0, 0.0, 1.0]
    centers_y = [-1.0, 0.0, 1.0]
    
    # Initialize prediction with the constant term
    pred = np.full_like(x, params[0], dtype=np.float64)
    
    # Add RBF contributions
    idx = 1
    for cx in centers_x:
        for cy in centers_y:
            # Compute squared distance
            dx = x - cx
            dy = y - cy
            dist_sq = dx**2 + dy**2
            
            # Compute RBF value
            rbf_val = np.exp(-gamma * dist_sq)
            
            # Add weighted contribution
            pred += params[idx] * rbf_val
            idx += 1
            
    return pred
# EVOLVE-BLOCK-END
