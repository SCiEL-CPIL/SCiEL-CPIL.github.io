import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Linear + Non-linear (Tanh) model.
    params:
    p0: Linear bias
    p1, p2, p3: Linear weights for x, y, z
    p4: Scale weight for the non-linear term
    p5, p6, p7: Weights for the non-linear term inputs
    p8: Bias for the non-linear term
    p9: Additional constant offset (effectively output bias)
    
    Model:
    pred = p0 + p1*x + p2*y + p3*z + p4 * tanh(p5*x + p6*y + p7*z + p8) + p9
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Linear part
    linear_part = p0 + p1 * x + p2 * y + p3 * z
    
    # Non-linear part
    inner = p5 * x + p6 * y + p7 * z + p8
    non_linear_part = p4 * np.tanh(inner)
    
    # Combine
    pred = linear_part + non_linear_part + p9
    
    return pred
