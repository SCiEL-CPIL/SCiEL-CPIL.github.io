"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: Bias + 3 Gaussian Radial Basis Functions.
    Parameters:
    p0: Bias
    p1, p2, p3: Weights for RBFs
    p4, p5, p6: Center offsets (added to base centers 0.5, 2.0, 4.0)
    p7, p8, p9: Log-widths (sigma = exp(p))
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Unpack
    bias = params[0]
    
    w1, w2, w3 = params[1], params[2], params[3]
    
    # Centers: Start at 1.0 (param) + Offset
    # Offsets chosen to spread initial centers across [0.1, 6]
    c1 = params[4] + 0.5
    c2 = params[5] + 2.0
    c3 = params[6] + 4.0
    
    # Log-widths: Start at 1.0 -> sigma = e^1 ~ 2.7
    s1, s2, s3 = params[7], params[8], params[9]
    
    # Ensure positive width and avoid numerical issues
    # Clip log-width to prevent exp overflow/underflow causing NaNs or 0s
    # s in [-10, 10] is safe. exp(10) ~ 22000, exp(-10) ~ 4.5e-5.
    s1 = np.clip(s1, -10, 10)
    s2 = np.clip(s2, -10, 10)
    s3 = np.clip(s3, -10, 10)
    
    sigma1 = np.exp(s1)
    sigma2 = np.exp(s2)
    sigma3 = np.exp(s3)
    
    # Calculate RBF terms
    # Using 0.5 * (x-c)^2 / sigma^2
    # To avoid division by zero if sigma is extremely small (though exp(-10) is not zero)
    rbf1 = np.exp(-0.5 * ((x - c1) / sigma1) ** 2)
    rbf2 = np.exp(-0.5 * ((x - c2) / sigma2) ** 2)
    rbf3 = np.exp(-0.5 * ((x - c3) / sigma3) ** 2)
    
    return bias + w1 * rbf1 + w2 * rbf2 + w3 * rbf3
# EVOLVE-BLOCK-END
