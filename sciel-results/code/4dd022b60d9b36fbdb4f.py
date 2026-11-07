import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # RBF centers and widths
    centers = np.array([-2.5, -1.25, 0.0, 1.25, 2.5])
    sigma = 0.8
    
    # Compute RBF terms
    rbf_terms = np.zeros_like(x)
    for i in range(5):
        rbf_terms += p[i] * np.exp(-0.5 * ((x - centers[i]) / sigma) ** 2)
    
    # Polynomial terms for global trend
    poly_terms = (p[5] + 
                  p[6] * x + 
                  p[7] * x**2 + 
                  p[8] * x**3 + 
                  p[9] * x**4)
                  
    return rbf_terms + poly_terms
