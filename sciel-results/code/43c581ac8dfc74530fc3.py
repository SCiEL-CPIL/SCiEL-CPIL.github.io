import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # z-score
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Probability of Improvement
    # PoI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    poi = norm.cdf(z)
    
    # Score: PoI per unit cost
    scores = poi / cost_safe
    
    # Ensure finite values
    scores[~np.isfinite(scores)] = -np.inf
    
    return scores
