import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    eps = 1e-9
    
    # Expected Improvement
    z = (f_best - mu) / (sigma + eps)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus proportional to sigma to encourage
    # visiting high-uncertainty regions (helps escape local minima in
    # the multimodal Levy landscape). The term is scaled so that it is
    # comparable to EI when mu is near f_best.
    # Use a modest weight to avoid over-exploration.
    exploration_weight = 0.1
    scores = (ei + exploration_weight * sigma) / (cost + eps)
    
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    return scores
