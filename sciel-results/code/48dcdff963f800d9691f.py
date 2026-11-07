"""
Symbolic regression using a mixed basis of polynomial, trigonometric, and hyperbolic functions.
This approach models y as a linear combination of 10 diverse basis functions.
The inclusion of trigonometric and hyperbolic terms allows the model to capture
periodic, saturating, and exponential behaviors that a pure polynomial basis
might approximate poorly, especially if the target function contains such features.
Since the model is linear in the parameters, the optimization problem remains convex,
ensuring reliable convergence of scipy.optimize.minimize (BFGS) to the global minimum.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points in [-2, 2]
    params: array of 10 coefficients
    Returns: predicted values
    """
    # Ensure x is a 1D array for consistent broadcasting if needed, 
    # though numpy handles broadcasting well.
    
    # Define the 10 basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2-5. Polynomial terms
    b1 = x
    b2 = x**2
    b3 = x**3
    b4 = x**4
    
    # 6-7. Trigonometric terms
    b5 = np.sin(x)
    b6 = np.cos(x)
    
    # 8. Exponential term
    b7 = np.exp(x)
    
    # 9. Hyperbolic tangent (saturation)
    b8 = np.tanh(x)
    
    # 10. Lorentzian (smooth peak)
    b9 = 1.0 / (1.0 + x**2)
    
    # Stack basis functions into a design matrix
    # Shape: (N, 10)
    # Each column is a basis function evaluated at all points
    V = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    # Compute predictions: V @ params
    # params is shape (10,)
    pred = V.dot(params)
    
    return pred
# EVOLVE-BLOCK-END
