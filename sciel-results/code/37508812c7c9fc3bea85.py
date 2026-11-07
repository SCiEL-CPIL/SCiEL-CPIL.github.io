import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a normalized monomial basis tailored for nonlinear dynamics.
    Inputs are normalized to improve conditioning for the optimizer.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to approximately [-1, 1] or zero-mean unit-variance
    # x range: [-0.5728, 0.5275] -> center ~0, scale ~0.55
    # v range: [-0.1754, 0.2821] -> center ~0.05, scale ~0.23
    
    X = x / 0.55
    V = (v - 0.05) / 0.23
    
    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    X2V = X2 * V
    X3V = X3 * V
    X2V2 = X2 * V2
    
    # Basis terms:
    # 0: Constant
    # 1: Linear X
    # 2: Linear V
    # 3: Quadratic X
    # 4: Cubic X
    # 5: Quadratic V
    # 6: Cross XV
    # 7: Cross X2V
    # 8: Cross X V2
    # 9: Cross X3V (captures nonlinearity in damping or stiffness coupling)
    
    term0 = 1.0
    term1 = X
    term2 = V
    term3 = X2
    term4 = X3
    term5 = V2
    term6 = XV
    term7 = X2V
    term8 = X * V2
    term9 = X3V
    
    return (p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 + p4 * term4 +
            p5 * term5 + p6 * term6 + p7 * term7 + p8 * term8 + p9 * term9)
