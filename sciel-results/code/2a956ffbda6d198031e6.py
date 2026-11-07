import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Probability of Improvement (PI) for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # We maximize this probability
    z = (f_best - mu) / sigma
    scores = norm.cdf(z)
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
