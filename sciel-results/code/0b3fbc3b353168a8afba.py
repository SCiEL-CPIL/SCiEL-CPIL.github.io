import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # 3 Radial Basis Functions + 1 Bias = 10 parameters
    # p[0]: bias
    # p[1], p[2], p[3]: centers
    # p[4], p[5], p[6]: widths (squared to ensure positivity)
    # p[7], p[8], p[9]: weights
    
    bias = p[0]
    
    c1, c2, c3 = p[1], p[2], p[3]
    # Add small epsilon to prevent division by zero if p is exactly 0
    s1 = p[4]**2 + 1e-6
    s2 = p[5]**2 + 1e-6
    s3 = p[6]**2 + 1e-6
    
    w1, w2, w3 = p[7], p[8], p[9]
    
    # Compute RBFs: exp(-(x-c)^2 / (2*s^2))
    # Note: s is already squared in the formula derivation if we consider standard Gaussian
    # Standard Gaussian: exp(-0.5 * ((x-mu)/sigma)^2) = exp(-(x-mu)^2 / (2*sigma^2))
    # Here s = sigma^2. So denominator is 2*s.
    
    rbf1 = np.exp(-((x - c1)**2) / (2.0 * s1))
    rbf2 = np.exp(-((x - c2)**2) / (2.0 * s2))
    rbf3 = np.exp(-((x - c3)**2) / (2.0 * s3))
    
    return bias + w1 * rbf1 + w2 * rbf2 + w3 * rbf3
