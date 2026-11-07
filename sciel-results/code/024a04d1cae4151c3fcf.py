# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using 10 RBFs at fixed, evenly spaced centers
    with a single learnable width parameter shared by all kernels.
    
    params layout (length 10):
      p[0]: shared width sigma (clamped to [0.05, 10.0] for stability)
      p[1..9]: coefficients for the 9 RBFs
    
    The 10th parameter is used as an implicit bias by including a constant
    RBF-like term, but actually we use 9 RBFs + 1 width = 10 params.
    Wait, that gives only 9 coefficients. Let me restructure:
    
    Better: use 10 RBFs at fixed centers, with p[0] as width and p[1..9] as 
    9 coefficients... but that's only 9 coefficients for 10 RBFs.
    
    Let's use: p[0] = width, p[1..10] would be 10 coefficients but we only have
    10 total params. So: p[0] = width, p[1..9] = 9 coefficients for 9 RBFs,
    and the 10th RBF coefficient is implicitly 1.0? No, that's not clean.
    
    Alternative: Use 10 RBFs, with p[0] as width and p[1..9] as coefficients,
    and add a constant term (bias) as the 10th basis function. So:
      - 9 RBFs with shared width p[0]
      - 1 constant term (bias) with coefficient p[9]
    
    Actually, let's keep it simple: 10 RBFs at fixed centers, with a single
    shared width. We need 10 coefficients + 1 width = 11 params, but we only
    have 10. So let's use 9 RBFs + 1 width + 1 bias = 11... still too many.
    
    Final design: 9 RBFs at fixed centers + 1 shared width = 10 params.
    The bias is captured by the RBFs themselves (they can approximate constants
    when width is large).
    
    Centers: evenly spaced in [-3, 3]: c_i = -3 + 6*i/8 for i=0..8
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    
    # Extract width and clamp for numerical stability
    sigma = params[0]
    sigma = np.clip(sigma, 0.05, 10.0)
    sigma2 = sigma * sigma
    
    # Fixed centers: 9 evenly spaced points in [-3, 3]
    centers = np.linspace(-3.0, 3.0, 9)
    
    # Coefficients for the 9 RBFs
    coeffs = params[1:10]
    
    # Compute RBF values: N x 9
    # diff[i, j] = x[i] - centers[j]
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf_values = np.exp(-0.5 * (diff / sigma2) ** 2)
    
    # Linear combination
    result = rbf_values @ coeffs
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
