"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y from x using a linear combination of 10 well-conditioned basis functions.
    The basis is a mix of low-degree polynomials (scaled for stability) and trigonometric
    functions to cover both non-periodic and periodic behaviors on [-3, 3].
    
    The scaling of polynomial terms helps BFGS by keeping the gradient magnitudes
    comparable across terms, improving convergence stability.
    """
    # Ensure x is a 1D float array
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Pad params if necessary (though evaluator should provide 10)
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    
    # Basis functions
    # We use a mix of:
    # 0: Constant
    # 1-3: Low-degree polynomials (scaled by 1/3 to keep values in O(1) range on [-3,3])
    # 4-9: Trigonometric functions (sin/cos at fundamental and 2nd harmonic)
    
    # Scaling factor for polynomials to improve conditioning
    # x is in [-3, 3], so x/3 is in [-1, 1]
    t = x / 3.0
    
    # 0: Constant
    b0 = np.ones_like(x)
    
    # 1: Linear (scaled)
    b1 = t
    
    # 2: Quadratic (scaled)
    b2 = t * t
    
    # 3: Cubic (scaled)
    b3 = t * t * t
    
    # 4: sin(x)
    b4 = np.sin(x)
    
    # 5: cos(x)
    b5 = np.cos(x)
    
    # 6: sin(2x)
    b6 = np.sin(2.0 * x)
    
    # 7: cos(2x)
    b7 = np.cos(2.0 * x)
    
    # 8: sin(3x) - adding 3rd harmonic for better periodic approximation
    b8 = np.sin(3.0 * x)
    
    # 9: cos(3x)
    b9 = np.cos(3.0 * x)
    
    # Linear combination
    # The parameters p[0..9] will be fitted by BFGS.
    # Because the basis functions are well-scaled (polynomials in [-1,1] range, trig in [-1,1]),
    # the loss landscape should be relatively smooth for BFGS.
    
    y = (p[0] * b0 +
         p[1] * b1 +
         p[2] * b2 +
         p[3] * b3 +
         p[4] * b4 +
         p[5] * b5 +
         p[6] * b6 +
         p[7] * b7 +
         p[8] * b8 +
         p[9] * b9)
    
    return y

# EVOLVE-BLOCK-END
