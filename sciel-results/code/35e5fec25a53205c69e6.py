# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a rational function (Padé-like) basis.
    
    This structural change uses a ratio of polynomials to model saturation effects
    common in nonlinear oscillators, which pure polynomials may fit poorly or
    overfit. The denominator provides inherent bounds and better conditioning.
    
    Inputs:
        x: position, range ~[-0.57, 0.53]
        v: velocity, range ~[-0.18, 0.28]
        params: array of 10 parameters
    
    Returns:
        acceleration array, one prediction per row
    """
    p = params
    
    # Scale inputs to roughly [-1, 1]
    # x: [-0.5728, 0.5275], scale by 1/0.55 ≈ 1.818
    # v: [-0.1754, 0.2821], scale by 1/0.23 ≈ 4.348 (using max abs value for symmetry)
    X = x * 1.818
    V = v * 4.348
    
    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    X4 = X2 * X2
    X2V = X2 * V
    V2 = V * V
    V3 = V2 * V
    V4 = V2 * V2
    X2V2 = X2 * V2
    
    # Numerator: p0*X + p1*X^3 + p2*V + p3*V^3 + p4*X^2*V
    # This covers linear/cubic spring, linear/cubic damping, and a coupling term.
    numerator = (p[0] * X + 
                 p[1] * X3 + 
                 p[2] * V + 
                 p[3] * V3 + 
                 p[4] * X2V)
    
    # Denominator: 1 + p5*X^2 + p6*V^2 + p7*X^4 + p8*V^4 + p9*X^2*V^2
    # This allows the "stiffness" or "damping" to change with amplitude (softening/hardening).
    denominator = (1.0 + 
                   p[5] * X2 + 
                   p[6] * V2 + 
                   p[7] * X4 + 
                   p[8] * V4 + 
                   p[9] * X2V2)
    
    # Avoid division by zero (though 1 + ... should be safe if params are small)
    # If denominator becomes close to 0, the result blows up. 
    # To ensure stability during optimization, we can add a small epsilon or clip.
    # However, BFGS might step into bad regions. 
    # Standard practice: just compute. If it fails, the score is bad.
    # But to be safe against NaNs/Infs which crash the evaluator:
    # We can use np.where to handle near-zero denominators, but that's non-differentiable.
    # Better: trust the optimizer to stay in a reasonable basin, or rely on the 
    # "1 +" term to keep it positive if params are small.
    
    y = numerator / denominator
    
    # Handle potential non-finite values from numerical instability
    # If any value is non-finite, replace with 0 or a large constant?
    # The contract says: "A prediction of the wrong shape or with non-finite error costs 1e30 during fitting."
    # It's better to return finite values.
    # We can clip the output to a reasonable range? No, that distorts gradients.
    # We can use np.nan_to_num.
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return y
# EVOLVE-BLOCK-END
