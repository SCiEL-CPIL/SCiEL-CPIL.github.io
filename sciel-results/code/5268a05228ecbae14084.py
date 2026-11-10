"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of a hybrid basis.
    
    The basis combines:
    1. Polynomial terms (degree 0-4) for global trends.
    2. Tanh terms for bounded non-linearities.
    3. Gaussian terms for localized features.
    4. A sinusoidal term for oscillatory behavior.
    
    This linear-in-parameters model ensures the MSE landscape is convex,
    allowing BFGS to reliably converge to the global minimum.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Ensure x is within a safe range to prevent overflow in exp/sin, 
    # though inputs are specified as [-2, 2].
    x_safe = np.clip(x, -10, 10)
    
    # Precompute powers
    x2 = x_safe * x_safe
    x3 = x2 * x_safe
    x4 = x2 * x2
    
    # Non-linear features
    tanh_x = np.tanh(x_safe)
    exp_neg_x2 = np.exp(-x2)
    exp_neg_half_x2 = np.exp(-0.5 * x2)
    sin_x = np.sin(x_safe)
    
    # Define the 10 basis functions
    # b0: Constant
    b0 = np.ones_like(x_safe)
    # b1: Linear
    b1 = x_safe
    # b2: Quadratic
    b2 = x2
    # b3: Cubic
    b3 = x3
    # b4: Quartic
    b4 = x4
    # b5: Tanh (odd symmetry)
    b5 = tanh_x
    # b6: x * tanh(x) (even symmetry, captures curvature)
    b6 = x_safe * tanh_x
    # b7: Gaussian centered at 0
    b7 = exp_neg_x2
    # b8: Wider Gaussian centered at 0
    b8 = exp_neg_half_x2
    # b9: Sine wave
    b9 = sin_x
    
    # Linear combination
    # Using dot product for efficiency and clarity
    # Construct basis matrix (N, 10)
    basis = np.column_stack((b0, b1, b2, b3, b4, b5, b6, b7, b8, b9))
    
    result = basis @ p
    
    # Safety check for non-finite values
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result

# EVOLVE-BLOCK-END
