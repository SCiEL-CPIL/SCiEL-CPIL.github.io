"""
Strategy: Linear-in-Parameters Basis with Fixed RBFs and Polynomial Backbone.
Instead of optimizing the centers of RBFs (non-convex, hard for BFGS), we fix a set of
RBF centers and polynomial terms. The 10 parameters are linear weights.
This makes the optimization problem convex (least squares), which BFGS solves very reliably
and quickly. The basis is designed to cover the domain [-1, 1]^2 well.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a linear combination of fixed basis functions.
    
    Basis functions (10 total):
    1. Constant: 1
    2. Linear x: x
    3. Linear y: y
    4. Quadratic x^2
    5. Quadratic y^2
    6. Interaction xy
    7. RBF center (0, 0), sigma=0.8
    8. RBF center (0.5, 0.5), sigma=0.5
    9. RBF center (-0.5, -0.5), sigma=0.5
    10. RBF center (0.5, -0.5), sigma=0.5
    
    params: [w0, w1, ..., w9]
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    params = np.asarray(params, dtype=float).flatten()
    
    # Safety check for params length
    if params.size != 10:
        # If params are wrong, return zeros to avoid crashing, though evaluator expects 10
        return np.zeros_like(x, dtype=float)

    # Extract weights
    w = params
    
    # 1. Polynomial Basis (Global structure)
    # These capture low-frequency trends and ensure the function is well-behaved at boundaries
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = x**2
    b4 = y**2
    b5 = x * y
    
    # 2. RBF Basis (Local structure)
    # Fixed centers spread across the domain [-1, 1] x [-1, 1]
    # Sigma values chosen to be broad enough to overlap but distinct enough to localize
    
    # Center (0, 0) - broad
    dist0_sq = x**2 + y**2
    b6 = np.exp(-dist0_sq / (0.8**2))
    
    # Center (0.5, 0.5) - medium
    dist1_sq = (x - 0.5)**2 + (y - 0.5)**2
    b7 = np.exp(-dist1_sq / (0.5**2))
    
    # Center (-0.5, -0.5) - medium
    dist2_sq = (x + 0.5)**2 + (y + 0.5)**2
    b8 = np.exp(-dist2_sq / (0.5**2))
    
    # Center (0.5, -0.5) - medium
    dist3_sq = (x - 0.5)**2 + (y + 0.5)**2
    b9 = np.exp(-dist3_sq / (0.5**2))
    
    # Linear combination
    pred = (
        w[0] * b0 +
        w[1] * b1 +
        w[2] * b2 +
        w[3] * b3 +
        w[4] * b4 +
        w[5] * b5 +
        w[6] * b6 +
        w[7] * b7 +
        w[8] * b8 +
        w[9] * b9
    )
    
    # Ensure finite output
    return np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
