import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a monomial polynomial basis.
    The model uses 10 terms: all polynomials of degree <= 2, plus x^3, y^3, x^2y, and x^2y^2.
    This includes the (2,2) interaction term which is missing in the standard degree-3 complete polynomial,
    potentially capturing stronger quadratic interaction effects in the hidden function.
    """
    # 10 monomial basis functions
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x**2
    b4 = y**2
    b5 = x * y
    b6 = x**3
    b7 = y**3
    b8 = x**2 * y
    b9 = x**2 * y**2
    
    # Stack into matrix of shape (N, 10)
    B = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Prediction is B @ params
    pred = B.dot(params)
    
    return pred
