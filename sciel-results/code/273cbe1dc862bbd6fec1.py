import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled with a sparse physical basis
    including higher-order stiffness and damping terms (up to 5th order).
    Structure: K(x)*x + D(v)*v + Cross terms + Bias
    """
    # Scale inputs to keep powers O(1) in the training range.
    # x ~ 0.5, v ~ 0.3
    X = x / 0.5
    V = v / 0.3
    
    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    X5 = X4 * X
    
    V2 = V * V
    V3 = V2 * V
    V4 = V2 * V2
    V5 = V4 * V
    
    # Cross terms
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2
    
    # 
    # p0: Bias (constant acceleration, likely near 0)
    # p1: Linear Stiffness (X)
    # p2: Cubic Stiffness (X^3)
    # p3: Quintic Stiffness (X^5)
    # p4: Linear Damping (V)
    # p5: Cubic Damping (V^3)
    # p6: Quintic Damping (V^5)
    # p7: Cross term X*V
    # p8: Cross term X^2*V
    # p9: Cross term X*V^2
    
    a = (params[0]
         + params[1] * X
         + params[2] * X3
         + params[3] * X5
         + params[4] * V
         + params[5] * V3
         + params[6] * V5
         + params[7] * XV
         + params[8] * X2V
         + params[9] * XV2)
         
    return a
