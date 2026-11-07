import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Polynomial Basis approach
    # Domain x is in [-2, 2]
    # Scale x to u in [-1, 1] for numerical stability
    # We use 10 parameters for a polynomial of degree 9
    # p[0]: constant
    # p[1]: u
    # ...
    # p[9]: u^9
    
    n = len(x)
    u = x / 2.0
    
    # Construct Vandermonde-like matrix
    # Using np.vander with increasing powers
    # np.vander(u, N=10, increasing=True) gives [1, u, u^2, ..., u^9]
    
    basis = np.vander(u, N=10, increasing=True)
    
    return basis @ params
