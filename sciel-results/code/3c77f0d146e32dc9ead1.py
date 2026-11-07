import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to approximately [-1, 1] to improve numerical conditioning
    # Estimated ranges: x ~ [-0.6, 0.6], v ~ [-0.3, 0.3]
    X = x / 0.6
    V = v / 0.3
    
    # Use trigonometric and polynomial terms to capture nonlinear oscillator dynamics
    # Trigonometric terms for periodic behavior
    sin_X = np.sin(X)
    cos_X = np.cos(X)
    sin_V = np.sin(V)
    
    # Polynomial terms
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    XV = X * V
    X2V = X2 * V
    
    # Combine terms with fitted parameters
    # Structure: constant, linear, damping, nonlinear restoring, cross terms
    return (p0 + p1 * X + p2 * V + p3 * sin_X + p4 * cos_X - 1.0 + 
            p5 * V2 + p6 * sin_V + p7 * XV + p8 * X2V + p9 * X3)
