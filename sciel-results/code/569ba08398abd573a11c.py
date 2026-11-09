"""Symbolic regression: Hybrid Polynomial + Learnable Center RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Quadratic Polynomial + 2 Gaussian RBFs with learnable centers and widths.
    
    10 parameters total:
      p[0]: Polynomial coefficient for 1 (part of bias handling, or just a0)
      p[1]: Polynomial coefficient for x
      p[2]: Polynomial coefficient for x^2
      p[3]: Center of RBF 1
      p[4]: Width (inverse scale) of RBF 1
      p[5]: Weight of RBF 1
      p[6]: Center of RBF 2
      p[7]: Width (inverse scale) of RBF 2
      p[8]: Weight of RBF 2
      p[9]: Global Bias
      
    The polynomial captures global trends, while the RBFs capture local deviations.
    Learnable centers allow the model to adapt to the specific features of the hidden function.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Extract parameters
    a0 = p[0]
    a1 = p[1]
    a2 = p[2]
    
    c1 = p[3]
    s1 = p[4]
    w1 = p[5]
    
    c2 = p[6]
    s2 = p[7]
    w2 = p[8]
    
    bias = p[9]
    
    # Compute polynomial part
    # Using Horner's method or direct expansion for stability
    poly_part = a0 + a1 * x + a2 * (x * x)
    
    # Compute RBF parts
    # Use abs(sigma) + epsilon to ensure positive width and avoid division by zero
    # This makes the function differentiable almost everywhere and robust to sign flips
    eps = 1e-6
    sigma1 = np.abs(s1) + eps
    sigma2 = np.abs(s2) + eps
    
    # Gaussian: exp(-0.5 * ((x - c) / sigma)^2)
    # Using (x-c)^2 / (2*sigma^2) is standard
    diff1 = x - c1
    rbf1 = np.exp(-(diff1 * diff1) / (2.0 * sigma1 * sigma1))
    
    diff2 = x - c2
    rbf2 = np.exp(-(diff2 * diff2) / (2.0 * sigma2 * sigma2))
    
    # Combine
    y = poly_part + w1 * rbf1 + w2 * rbf2 + bias
    
    return y
# EVOLVE-BLOCK-END
