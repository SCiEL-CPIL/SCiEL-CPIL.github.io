import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Quantile-based Probability of Improvement (POI) acquisition function.
    Computes P(f < f_best) for each candidate, where f ~ N(mu, sigma^2).
    Higher probability of improvement means higher score.
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Standardize: z = (f_best - mu) / sigma
    # P(f < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -10, 10)
    
    # CDF of standard normal gives probability of improvement
    scores = norm.cdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
