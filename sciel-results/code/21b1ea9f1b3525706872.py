"""RBF basis expansion for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum of RBF (Gaussian) basis functions:
    y = a0 + sum_{i=1}^{5} a_i * exp(-beta_i * (x - c_i)^2)
    
    params layout (length 10):
      params[0] = a0 (constant offset)
      params[1] = a1, params[2] = c1, params[3] = beta1
      params[4] = a2, params[5] = c2, params[6] = beta2
      params[7] = a3, params[8] = c3, params[9] = beta3
    
    Wait, that's 1 + 3*3 = 10 params. But we want 5 RBFs.
    Let's use:
      params[0] = a0 (constant)
      params[1..9] = 3 params per RBF for 3 RBFs: (a_i, c_i, beta_i)
    
    Actually, let's do 4 RBFs + 1 constant = 13 params, too many.
    Let's do 3 RBFs + 1 constant + 2 extra = 12, still too many.
    
    Better: 5 RBFs with shared structure? No, let's just use 3 RBFs + constant.
    params[0] = a0
    params[1] = a1, params[2] = c1, params[3] = beta1
    params[4] = a2, params[5] = c2, params[6] = beta2
    params[7] = a3, params[8] = c3, params[9] = beta3
    
    That's exactly 10 params.
    
    beta_i is constrained positive via softplus: beta_i = log(1 + exp(params[2i+2]))
    c_i is centered in [-2, 2] via tanh: c_i = 2 * tanh(params[2i+1])
    a_i is free.
    """
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    a0 = params[0]
    
    # 3 RBF terms
    # Term 1: a1, c1, beta1
    a1 = params[1]
    c1_raw = params[2]
    beta1_raw = params[3]
    
    # Term 2: a2, c2, beta2
    a2 = params[4]
    c2_raw = params[5]
    beta2_raw = params[6]
    
    # Term 3: a3, c3, beta3
    a3 = params[7]
    c3_raw = params[8]
    beta3_raw = params[9]
    
    # Transform centers to [-2, 2]
    c1 = 2.0 * np.tanh(c1_raw)
    c2 = 2.0 * np.tanh(c2_raw)
    c3 = 2.0 * np.tanh(c3_raw)
    
    # Transform widths to positive via softplus
    def softplus(v):
        # Numerically stable softplus
        return np.where(v > 30, v, np.log1p(np.exp(v)))
    
    beta1 = softplus(beta1_raw)
    beta2 = softplus(beta2_raw)
    beta3 = softplus(beta3_raw)
    
    # Compute RBF terms
    # Use a minimum beta to avoid division by zero or extremely narrow peaks
    beta_min = 1e-4
    beta1 = np.maximum(beta1, beta_min)
    beta2 = np.maximum(beta2, beta_min)
    beta3 = np.maximum(beta3, beta_min)
    
    d1 = x - c1
    d2 = x - c2
    d3 = x - c3
    
    rbf1 = np.exp(-beta1 * d1 * d1)
    rbf2 = np.exp(-beta2 * d2 * d2)
    rbf3 = np.exp(-beta3 * d3 * d3)
    
    result = a0 + a1 * rbf1 + a2 * rbf2 + a3 * rbf3
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
