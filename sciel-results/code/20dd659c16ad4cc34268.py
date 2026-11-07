import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement
    with a small exploration bonus.
    """
    sigma = np.maximum(sigma, 1e-6)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Small exploration bonus to avoid premature convergence
    exploration = 0.1 * sigma
    scores = (ei + exploration) / cost
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
