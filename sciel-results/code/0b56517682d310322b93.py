import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using a sum of Gaussian bumps.
    f(x) = C + sum_{i=1}^{3} A_i * exp(-(x - mu_i)^2 / (2 * sigma_i^2))
    params: [A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, C]
    """
    A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, C = params
    
    # Ensure sigma is not zero to avoid division by zero.
    # Using square ensures positivity. Adding epsilon prevents 0.
    eps = 1e-6
    s1 = sigma1 * sigma1 + eps
    s2 = sigma2 * sigma2 + eps
    s3 = sigma3 * sigma3 + eps
    
    # Calculate each Gaussian term
    # Using np.exp with clipped arguments might be safer if overflow is a concern,
    # but for x in [-2, 2] and reasonable params, it should be fine.
    # If (x-mu) is large and sigma is small, exp goes to 0.
    # If (x-mu) is 0, exp is 1.
    
    g1 = A1 * np.exp(-0.5 * (x - mu1) ** 2 / s1)
    g2 = A2 * np.exp(-0.5 * (x - mu2) ** 2 / s2)
    g3 = A3 * np.exp(-0.5 * (x - mu3) ** 2 / s3)
    
    y = C + g1 + g2 + g3
    
    # Ensure finite output
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
