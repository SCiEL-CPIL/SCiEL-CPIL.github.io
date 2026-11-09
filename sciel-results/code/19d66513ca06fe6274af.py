import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using an orthogonal polynomial basis model.
    The model is a linear combination of 10 2D Legendre polynomial basis functions
    spanning constants, linear, quadratic, and cubic terms in x and y.
    """
    # Compute 1D Legendre polynomials up to degree 3
    # P0 = 1
    # P1 = x
    # P2 = 0.5 * (3x^2 - 1)
    # P3 = 0.5 * (5x^3 - 3x)
    
    P0_x = np.ones_like(x)
    P1_x = x
    P2_x = 0.5 * (3 * x**2 - 1)
    P3_x = 0.5 * (5 * x**3 - 3 * x)
    
    P0_y = np.ones_like(y)
    P1_y = y
    P2_y = 0.5 * (3 * y**2 - 1)
    P3_y = 0.5 * (5 * y**3 - 3 * y)
    
    # Construct 2D basis functions (products of 1D Legendre polynomials)
    # Total 10 terms covering degrees 0 to 3
    b0 = P0_x * P0_y
    b1 = P1_x * P0_y
    b2 = P0_x * P1_y
    b3 = P2_x * P0_y
    b4 = P0_x * P2_y
    b5 = P1_x * P1_y
    b6 = P3_x * P0_y
    b7 = P0_x * P3_y
    b8 = P2_x * P1_y
    b9 = P1_x * P2_y
    
    # Stack into matrix of shape (N, 10)
    B = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Prediction is B @ params
    pred = B.dot(params)
    
    return pred
