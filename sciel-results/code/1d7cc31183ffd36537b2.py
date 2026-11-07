import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of a diverse set of basis functions.
    
    The model is:
        f(x) = sum_{i=0}^{9} params[i] * basis_i(x)
    
    Basis functions:
        0: 1
        1: x
        2: x^2
        3: x^3
        4: x^4
        5: sin(x)
        6: cos(x)
        7: exp(-x^2)
        8: tanh(x)
        9: 1 / (1 + x^2)
    
    This model is linear in parameters, ensuring a convex loss landscape
    for the BFGS optimizer, which should lead to reliable convergence
    to the global minimum of the training MSE.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Compute basis functions
    # Using np.column_stack for efficiency
    # Note: x is 1D array of shape (N,)
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = b2 * x
    b4 = b2 * b2
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.exp(-b2)
    b8 = np.tanh(x)
    b9 = 1.0 / (1.0 + b2)
    
    # Stack into a matrix of shape (N, 10)
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Linear combination
    y = basis @ params
    
    return y
