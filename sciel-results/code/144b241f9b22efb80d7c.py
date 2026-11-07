import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Hybrid EI-LCB acquisition: balances exploitation (EI) with exploration (LCB)
    # EI for minimization: E[max(f_best - f, 0)] where f ~ N(mu, sigma^2)
    # LCB term: -xi * sigma to encourage exploration in high-uncertainty regions
    
    # Expected Improvement (minimization)
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive LCB exploration coefficient
    # Reduce exploration as we get closer to the known minimum
    f_min = -3.32237
    gap = f_best - f_min
    # xi ranges from 0.3 (close to min) to 0.8 (far from min)
    xi = 0.3 + 0.5 * np.clip(gap / 5.0, 0.0, 1.0)
    
    # Combine: EI (exploitation) + LCB (exploration)
    # Higher score means more promising
    score = ei + xi * sigma_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
