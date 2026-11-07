import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Cubic polynomial model for symbolic regression.
    Basis functions:
    1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
    
    This model is linear in the parameters, ensuring that scipy.optimize.minimize
    (BFGS) converges to the global minimum (least squares solution) without getting
    stuck in local minima or encountering numerical instability from division by zero.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Construct the cubic polynomial
    # p0 * 1
    # p1 * x
    # p2 * y
    # p3 * x^2
    # p4 * y^2
    # p5 * x*y
    # p6 * x^3
    # p7 * x^2*y
    # p8 * x*y^2
    # p9 * y^3
    
    result = (p0 + 
              p1 * x + 
              p2 * y + 
              p3 * x**2 + 
              p4 * y**2 + 
              p5 * x * y + 
              p6 * x**3 + 
              p7 * x**2 * y + 
              p8 * x * y**2 + 
              p9 * y**3)
              
    return result
