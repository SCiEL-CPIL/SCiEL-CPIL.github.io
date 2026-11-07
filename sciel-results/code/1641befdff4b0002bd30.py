"""
Symbolic regression using a hybrid basis of Chebyshev polynomials and Gaussian RBFs.
- T0 to T4 (5 Chebyshev polynomials) for global smooth trends.
- 5 Gaussian RBFs centered at -3, -1.5, 0, 1.5, 3 for local variations.
This structure is well-conditioned and flexible.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Compute y = sum_i params[i] * phi_i(x)
    
    Basis functions:
    phi_0..4: Chebyshev polynomials T_0(x') to T_4(x') where x' = x/3 mapped to [-1,1]
    phi_5..9: Gaussian RBFs exp(-((x-c)^2)/(2*sigma^2)) with c in [-3, -1.5, 0, 1.5, 3]
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure we have exactly 10 params
    if params.shape[0] != 10:
        p = np.ones(10)
        n = min(10, params.shape[0])
        p[:n] = params[:n]
        params = p
    
    n_points = x.shape[0]
    phi = np.zeros((10, n_points))
    
    # 1. Chebyshev Polynomials (mapped to [-1, 1] for stability)
    # x is in [-3, 3], so x_mapped = x / 3 is in [-1, 1]
    x_map = x / 3.0
    
    # T_0 = 1
    phi[0, :] = 1.0
    # T_1 = x
    phi[1, :] = x_map
    # T_2 = 2x^2 - 1
    phi[2, :] = 2.0 * x_map**2 - 1.0
    # T_3 = 4x^3 - 3x
    phi[3, :] = 4.0 * x_map**3 - 3.0 * x_map
    # T_4 = 8x^4 - 8x^2 + 1
    phi[4, :] = 8.0 * x_map**4 - 8.0 * x_map**2 + 1.0
    
    # 2. Gaussian RBFs
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.0  # Width parameter
    
    for i, c in enumerate(centers):
        # Calculate distance squared
        diff = x - c
        # Clip argument to avoid overflow in exp, though with x in [-3,3] and c in [-3,3],
        # max diff is 6, diff^2 is 36, exp(-18) is small but finite.
        # We use 0.5 * sigma^2 in denominator.
        exponent = -0.5 * (diff**2) / (sigma**2)
        phi[5 + i, :] = np.exp(exponent)
        
    # y = params @ phi
    y = params @ phi
    
    # Ensure finite values
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
# EVOLVE-BLOCK-END
