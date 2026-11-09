# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a physics-informed basis.
    
    This model assumes the acceleration can be decomposed into position-dependent
    (restoring) and velocity-dependent (damping) terms, primarily odd functions,
    with even terms to capture asymmetry and a bias term.
    
    Inputs are scaled to approximately [-1, 1] to improve conditioning for BFGS.
    
    Inputs:
        x: position, range ~[-0.57, 0.53]
        v: velocity, range ~[-0.18, 0.28]
        params: array of 10 parameters
    
    Returns:
        acceleration array, one prediction per row
    """
    p = params
    
    # Scale inputs to roughly [-1, 1]
    # x range ~ [-0.57, 0.53], max abs ~ 0.57. Scale by 1/0.57 ~ 1.75
    # v range ~ [-0.18, 0.28], max abs ~ 0.28. Scale by 1/0.28 ~ 3.57
    X = x * 1.75
    V = v * 3.57
    
    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    X5 = X3 * X2
    
    V2 = V * V
    V3 = V2 * V
    V5 = V3 * V2
    
    # Basis functions:
    # Odd powers of X (Restoring force components)
    b1 = X
    b2 = X3
    b3 = X5
    
    # Odd powers of V (Damping components)
    b4 = V
    b5 = V3
    b6 = V5
    
    # Even powers (Asymmetry/Nonlinear corrections)
    b7 = X2
    b8 = V2
    
    # Interaction term
    b9 = X * V
    
    # Bias
    b10 = np.ones_like(X)
    
    # Combine
    y = (p[0] * b1 +
         p[1] * b2 +
         p[2] * b3 +
         p[3] * b4 +
         p[4] * b5 +
         p[5] * b6 +
         p[6] * b7 +
         p[7] * b8 +
         p[8] * b9 +
         p[9] * b10)
         
    return y
# EVOLVE-BLOCK-END
