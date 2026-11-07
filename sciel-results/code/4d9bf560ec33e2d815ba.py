"""Symbolic regression using a sum of shifted sigmoid basis functions."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum of shifted sigmoid basis functions:
    y = sum_{i=0}^{4} p_i / (1 + exp(-s_i * (x - c_i)))
    
    This models step-like, saturating, or piecewise-constant behaviors
    while maintaining global smoothness for BFGS convergence.
    
    params layout (length 10):
      params[0] = p_0 (amplitude for sigmoid 0)
      params[1] = c_0 (center for sigmoid 0)
      params[2] = s_0 (steepness for sigmoid 0)
      params[3] = p_1 (amplitude for sigmoid 1)
      params[4] = c_1 (center for sigmoid 1)
      params[5] = s_1 (steepness for sigmoid 1)
      params[6] = p_2 (amplitude for sigmoid 2)
      params[7] = c_2 (center for sigmoid 2)
      params[8] = s_2 (steepness for sigmoid 2)
      params[9] = p_3 (amplitude for sigmoid 3, with fixed center 0 and steepness 5)
    
    Note: To keep 10 params and have 4 sigmoids, we fix the last sigmoid's
    center and steepness to reasonable defaults. Actually, let's use 3 full
    sigmoids (9 params) + 1 constant term (1 param) = 10 params.
    
    Revised layout:
      params[0] = p_0 (amplitude for sigmoid 0)
      params[1] = c_0 (center for sigmoid 0)
      params[2] = s_0 (steepness for sigmoid 0)
      params[3] = p_1 (amplitude for sigmoid 1)
      params[4] = c_1 (center for sigmoid 1)
      params[5] = s_1 (steepness for sigmoid 1)
      params[6] = p_2 (amplitude for sigmoid 2)
      params[7] = c_2 (center for sigmoid 2)
      params[8] = s_2 (steepness for sigmoid 2)
      params[9] = b (constant offset)
    """
    x = np.asarray(x, dtype=float)
    
    if x.size == 0:
        return np.array([])
        
    # Extract parameters
    p0, c0, s0 = params[0], params[1], params[2]
    p1, c1, s1 = params[3], params[4], params[5]
    p2, c2, s2 = params[6], params[7], params[8]
    b = params[9]
    
    # Compute sigmoid terms with numerical stability
    # sigmoid(z) = 1 / (1 + exp(-z))
    # For large positive z, sigmoid ~ 1; for large negative z, sigmoid ~ 0
    
    def stable_sigmoid(z):
        """Numerically stable sigmoid."""
        # Use the identity: sigmoid(z) = 0.5 * (1 + tanh(z/2))
        # tanh is stable and bounded
        return 0.5 * (1.0 + np.tanh(0.5 * z))
    
    # Compute each sigmoid term
    z0 = s0 * (x - c0)
    z1 = s1 * (x - c1)
    z2 = s2 * (x - c2)
    
    sig0 = stable_sigmoid(z0)
    sig1 = stable_sigmoid(z1)
    sig2 = stable_sigmoid(z2)
    
    # Sum the terms
    result = p0 * sig0 + p1 * sig1 + p2 * sig2 + b
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return result
# EVOLVE-BLOCK-END
