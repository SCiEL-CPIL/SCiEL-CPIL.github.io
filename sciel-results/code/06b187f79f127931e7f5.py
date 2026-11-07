import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a 4th-order polynomial in strain with linear temperature dependence for each coefficient.
    sigma = sum_{k=0}^{4} (a_k + b_k * T) * eps^k
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    eps = strain
    T = temp
    
    # Coefficients as functions of Temperature
    # c_k(T) = a_k + b_k * T
    c0 = p0 + p1 * T
    c1 = p2 + p3 * T
    c2 = p4 + p5 * T
    c3 = p6 + p7 * T
    c4 = p8 + p9 * T
    
    # Horner's method for numerical stability and efficiency
    # sigma = c0 + eps * (c1 + eps * (c2 + eps * (c3 + eps * c4)))
    result = c0 + eps * (c1 + eps * (c2 + eps * (c3 + eps * c4)))
    
    return result
