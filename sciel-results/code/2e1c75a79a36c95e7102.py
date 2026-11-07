"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

# We define the basis functions here.
# The basis is a full polynomial of degree 3 in two variables.
# Terms: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
# This gives us 10 terms, matching the 10 parameters.

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a linear combination of 10 basis functions.
    The basis functions are the monomials of degree up to 3.
    """
    # Ensure inputs are float arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Compute basis terms
    # We construct the feature matrix F of shape (N, 10)
    # F[:, i] is the i-th basis function evaluated at the points
    
    # For efficiency and clarity, we compute them directly
    f0 = np.ones_like(x)
    f1 = x
    f2 = y
    f3 = x * x
    f4 = x * y
    f5 = y * y
    f6 = x * x * x
    f7 = x * x * y
    f8 = x * y * y
    f9 = y * y * y
    
    # Stack into a matrix (N, 10)
    # We can use np.column_stack or just dot product with a tuple
    # np.dot(p, [f0, f1, ...]) works if we construct the array properly
    
    # Let's construct the features matrix
    features = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))
    
    # Prediction is p @ features.T
    # p is (10,), features.T is (10, N) -> result (N,)
    preds = p @ features.T
    
    return preds
# EVOLVE-BLOCK-END
