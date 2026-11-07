"""
Strategy: Hybrid Polynomial-Gaussian Basis.

Rationale:
The previous cubic polynomial basis (score 3.5e8) suggests the hidden function
contains non-polynomial features (e.g., localized bumps, exponential decay, or
sharp transitions) that a global polynomial cannot capture efficiently without
very high degrees (which cause ill-conditioning).

This program uses a hybrid basis:
1. A 7-term polynomial backbone (up to x^3, y^2, xy) to capture global trends.
2. Two Gaussian Radial Basis Functions (RBFs) with fixed centers at (0,0) and 
   (0.5, 0.5) to capture local, non-polynomial features.
3. A shared, squared scale parameter for the RBFs to ensure numerical stability 
   (positive width) and smooth optimization landscape for BFGS.

Parameters (length 10):
p0: Offset
p1: Coeff x
p2: Coeff y
p3: Coeff x^2
p4: Coeff y^2
p5: Coeff xy
p6: Coeff x^3
p7: Amplitude of RBF at (0,0)
p8: Amplitude of RBF at (0.5, 0.5)
p9: Scale factor (used as p9^2 in exponent)
"""

# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Cubic-ish polynomial + 2 Gaussians.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial part
    # Note: We include x^3 but not y^3 to save a parameter for the 2nd RBF amplitude.
    # This asymmetry is a trade-off, but the RBFs provide some local flexibility.
    poly = (
        p0
        + p1 * x
        + p2 * y
        + p3 * x**2
        + p4 * y**2
        + p5 * x * y
        + p6 * x**3
    )
    
    # RBF part
    # Use p9^2 to ensure the variance/width is positive and the function is smooth w.r.t p9.
    scale = p9 * p9
    
    # RBF 1: Centered at (0, 0)
    rbf1_arg = scale * (x**2 + y**2)
    # Clip argument to prevent underflow/overflow issues in exp, though for [-1,1] and reasonable p9 it's safe.
    # exp(-large) -> 0. exp(-small) -> 1.
    rbf1 = np.exp(-rbf1_arg)
    
    # RBF 2: Centered at (0.5, 0.5)
    rbf2_arg = scale * ((x - 0.5)**2 + (y - 0.5)**2)
    rbf2 = np.exp(-rbf2_arg)
    
    # Combine
    pred = poly + p7 * rbf1 + p8 * rbf2
    
    return pred
# EVOLVE-BLOCK-END
