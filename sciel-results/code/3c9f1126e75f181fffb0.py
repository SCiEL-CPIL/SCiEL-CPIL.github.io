import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a physics-informed sparse basis.
    
    Inputs:
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
    
    Returns:
    Acceleration
    """
    # Scale inputs for better conditioning
    # x range ~ [-0.57, 0.53], center ~0, scale ~0.55
    # v range ~ [-0.18, 0.28], center ~0.05, scale ~0.25
    X = x / 0.55
    V = (v - 0.05) / 0.25
    
    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    XV = X * V
    X2V = X2 * V
    XV2 = X * V2
    
    # Trigonometric terms (common in oscillators)
    sinX = np.sin(X)
    cosX = np.cos(X)
    
    # Combine features with parameters
    # We have 10 parameters to assign
    # p0: linear x (restoring force)
    # p1: linear v (damping)
    # p2: quadratic x
    # p3: cubic x (Duffing nonlinearity)
    # p4: quadratic v
    # p5: cubic v
    # p6: cross term xv
    # p7: sin(x)
    # p8: cos(x) - 1 (to keep zero at x=0)
    # p9: constant offset
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    return (p0 * X + p1 * V + p2 * X2 + p3 * X3 + p4 * V2 + 
            p5 * V3 + p6 * XV + p7 * sinX + p8 * (cosX - 1.0) + p9)
