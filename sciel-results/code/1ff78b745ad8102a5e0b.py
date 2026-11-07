"""
Strategy: Hybrid Polynomial + Independent RBF Basis
This strategy combines a global polynomial trend with local radial basis functions.
Parameters:
- p[0:4]: Coefficients for polynomial terms 1, x, x^2, x^3
- p[4:6]: Centers for two independent RBFs
- p[6:8]: Inverse squared widths for the two RBFs (clipped to positive)
- p[8:10]: Amplitudes for the two RBFs
This allows the model to fit a cubic trend globally and correct for local variations 
with two flexible bumps, using all 10 parameters effectively.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y from x using a cubic polynomial plus two independent Gaussian RBFs.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
        
    # Extract parameters
    # p[0:4] -> Polynomial coefficients [c0, c1, c2, c3]
    c0, c1, c2, c3 = p[0], p[1], p[2], p[3]
    
    # p[4:6] -> Centers for RBFs
    c1_rbf = p[4]
    c2_rbf = p[5]
    
    # p[6:8] -> Inverse squared widths for RBFs
    # Clip to ensure positive width and numerical stability
    w1 = np.maximum(p[6], 0.1)
    w2 = np.maximum(p[7], 0.1)
    
    # p[8:10] -> Amplitudes for RBFs
    a1 = p[8]
    a2 = p[9]
    
    # Compute polynomial part
    poly = c0 + c1 * x + c2 * x**2 + c3 * x**3
    
    # Compute RBF part
    # RBF1: a1 * exp(-0.5 * w1 * (x - c1_rbf)^2)
    diff1 = x - c1_rbf
    rbf1 = a1 * np.exp(-0.5 * w1 * diff1**2)
    
    # RBF2: a2 * exp(-0.5 * w2 * (x - c2_rbf)^2)
    diff2 = x - c2_rbf
    rbf2 = a2 * np.exp(-0.5 * w2 * diff2**2)
    
    y = poly + rbf1 + rbf2
    
    return y

# EVOLVE-BLOCK-END
