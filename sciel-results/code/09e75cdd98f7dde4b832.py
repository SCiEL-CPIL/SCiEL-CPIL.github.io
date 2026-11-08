import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement (POI) per unit cost
    # POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # We want to maximize this, penalized by cost.
    
    # Handle sigma = 0
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    poi = norm.cdf(z)
    
    # Cost is 1 + mean(x), so it's >= 1
    # Divide by cost
    scores = poi / cost
    
    return scores
