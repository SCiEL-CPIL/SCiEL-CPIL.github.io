import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a linear combination of a fixed basis of 
    polynomials and Radial Basis Functions (RBFs).
    
    The model is linear in the 10 parameters, which allows scipy.optimize.minimize
    (BFGS) to efficiently find the global minimum of the Mean Squared Error.
    
    Basis functions:
    1. 1
    2. x
    3. y
    4. x^2
    5. y^2
    6. xy
    7. exp(-(x^2 + y^2))
    8. exp(-((x - 0.5)^2 + y^2))
    9. exp(-(x^2 + (y - 0.5)^2))
    10. exp(-((x + 0.5)^2 + (y + 0.5)^2))
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Flatten inputs
    x = x.flatten()
    y = y.flatten()
    n = x.shape[0]
    
    # Compute basis functions
    # Polynomials
    phi_1 = np.ones(n)
    phi_2 = x
    phi_3 = y
    phi_4 = x * x
    phi_5 = y * y
    phi_6 = x * y
    
    # RBFs (Gaussian kernels)
    # Center 1: (0, 0)
    phi_7 = np.exp(-(x * x + y * y))
    
    # Center 2: (0.5, 0)
    phi_8 = np.exp(-((x - 0.5) * (x - 0.5) + y * y))
    
    # Center 3: (0, 0.5)
    phi_9 = np.exp(-(x * x + (y - 0.5) * (y - 0.5)))
    
    # Center 4: (-0.5, -0.5)
    phi_10 = np.exp(-((x + 0.5) * (x + 0.5) + (y + 0.5) * (y + 0.5)))
    
    # Stack basis functions into a matrix of shape (n, 10)
    # Note: We use column_stack for efficiency
    basis = np.column_stack([
        phi_1, phi_2, phi_3, phi_4, phi_5, phi_6,
        phi_7, phi_8, phi_9, phi_10
    ])
    
    # Linear combination: pred = basis @ p
    pred = basis @ p
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
