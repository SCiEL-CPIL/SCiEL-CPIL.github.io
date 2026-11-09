import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid basis: 5 Gaussian RBFs + 5 Chebyshev polynomial terms
    # RBF centers: 4 corners + origin
    # Polynomial terms: 1, x, y, x*y, x^2+y^2 (Chebyshev-like)
    
    # Gaussian RBFs
    sigma = 0.5
    # Centers: (-1,-1), (-1,1), (1,-1), (1,1), (0,0)
    cx = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
    cy = np.array([-1.0, 1.0, -1.0, 1.0, 0.0])
    
    # Compute RBF features
    rbf_features = np.zeros((x.shape[0], 5))
    for i in range(5):
        dx = x - cx[i]
        dy = y - cy[i]
        rbf_features[:, i] = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
    
    # Polynomial basis (Chebyshev-inspired)
    poly_features = np.zeros((x.shape[0], 5))
    poly_features[:, 0] = 1.0  # constant
    poly_features[:, 1] = x
    poly_features[:, 2] = y
    poly_features[:, 3] = x * y
    poly_features[:, 4] = x**2 + y**2  # radial term
    
    # Combine: first 5 params for RBF, last 5 for polynomial
    pred = rbf_features @ params[:5] + poly_features @ params[5:]
    
    return pred
