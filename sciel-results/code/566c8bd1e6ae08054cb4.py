# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a physics-informed hybrid basis.
    
    Structure: a = -x * (k0 + k1*x^2 + k2*v) - v * (c0 + c1*x^2) + bias
    
    This is a Duffing-like oscillator with velocity-dependent stiffness correction
    and position-dependent damping. The basis is scaled so that optimal parameters
    are O(1), well-conditioned for BFGS starting from ones.
    
    Basis functions (10 terms):
    1. -x           (linear restoring force)
    2. -x^3         (Duffing nonlinear restoring)
    3. -x*v         (velocity-dependent stiffness / parametric)
    4. -v           (linear damping)
    5. -x^2*v       (position-modulated damping)
    6. x^2          (constant offset from position)
    7. v^2          (velocity quadratic correction)
    8. x^3*v        (higher-order nonlinear damping)
    9. x*v^2        (higher-order coupling)
    10. 1           (bias)
    
    Inputs:
        x: position, range ~[-0.57, 0.53]
        v: velocity, range ~[-0.18, 0.28]
        params: array of 10 parameters
    
    Returns:
        acceleration array, one prediction per row
    """
    p = params
    
    # Scale inputs to keep basis terms O(1)
    # x in [-0.5728, 0.5275], so x^2 in [0, 0.33], x^3 in [-0.19, 0.15]
    # v in [-0.1754, 0.2821], so v^2 in [0, 0.08]
    # These are already small, so we use moderate scaling
    X = x * 2.0   # x -> ~[-1.15, 1.06]
    V = v * 5.0   # v -> ~[-0.88, 1.41]
    
    # Precompute powers
    X2 = X * X
    X3 = X2 * X
    V2 = V * V
    V3 = V2 * V
    
    # Physics-informed basis functions
    # Restoring force terms (negative sign for restoring)
    b1 = -X
    b2 = -X3
    b3 = -X * V
    
    # Damping force terms (negative sign for damping)
    b4 = -V
    b5 = -X2 * V
    b8 = -X3 * V
    
    # Correction terms
    b6 = X2
    b7 = V2
    b9 = X * V2
    
    # Bias
    b10 = np.ones_like(X)
    
    # Combine with parameters
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
