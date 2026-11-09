import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Weighted Expected Improvement (EI)
    # Minimization problem
    
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # z-score for EI calculation
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Probability density function (pdf) and cumulative distribution function (cdf)
    # norm.pdf(z) = phi(z)
    # norm.cdf(z) = Phi(z)
    
    # EI formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, z is negative, Phi(z) is small, (f_best - mu) is negative.
    # The term (f_best - mu)*Phi(z) will be negative.
    # The term sigma*phi(z) is positive.
    # Standard EI is clipped at 0.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to 0 (if mu > f_best, EI should be 0)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    # We want to maximize this score
    scores = ei / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
