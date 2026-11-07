"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    
    Strategy: Use a physics-inspired Duffing-like polynomial basis, but 
    normalize the inputs x and v to have comparable magnitudes (approx [-1, 1]) 
    before computing the terms. This improves the conditioning of the Jacobian 
    for the BFGS optimizer, allowing it to converge to a better local minimum 
    than the unnormalized polynomial or simple linear model.
    """
    p = params
    
    # Normalize inputs based on training ranges provided in the prompt
    # x range: [-0.5728, 0.5275] -> scale by max abs ~0.57
    # v range: [-0.1754, 0.2821] -> scale by max abs ~0.28
    # We use fixed scales to avoid needing extra params for scaling.
    x_norm = x / 0.57
    v_norm = v / 0.28
    
    # Precompute powers and cross terms
    x2 = x_norm * x_norm
    x3 = x2 * x_norm
    v2 = v_norm * v_norm
    v3 = v2 * v_norm
    
    xv = x_norm * v_norm
    x2v = x2 * v_norm
    xv2 = x_norm * v2
    x3v = x3 * v_norm # Higher order interaction
    
    # Construct the equation using all 10 params
    # Basis:
    # 0: Constant (bias)
    # 1: Linear x
    # 2: Linear v (damping)
    # 3: Quadratic x (stiffness correction)
    # 4: Cubic x (Duffing nonlinearity)
    # 5: Quadratic v (nonlinear damping)
    # 6: Cross term x*v
    # 7: Cross term x^2*v
    # 8: Cross term x*v^2
    # 9: Cubic v (nonlinear damping)
    
    return (p[0] + 
            p[1] * x_norm + 
            p[2] * v_norm + 
            p[3] * x2 + 
            p[4] * x3 + 
            p[5] * v2 + 
            p[6] * xv + 
            p[7] * x2v + 
            p[8] * xv2 + 
            p[9] * v3)
# EVOLVE-BLOCK-END
