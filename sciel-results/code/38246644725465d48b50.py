"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB structure with EI refinement.
    
    The base structure is UCB/cost (which scored 1.0338), but we blend in an
    EI/cost term to leverage the probabilistic improvement estimate. The EI term
    naturally dampens exploration in regions where the GP mean is already below
    f_best, while still rewarding uncertainty in unexplored regions.
    
    Score = (alpha * UCB + (1 - alpha) * EI) / cost
    where UCB = (f_best - mu) + beta * sigma
    and EI = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu) / sigma
    
    We use alpha = 0.6 to weight UCB more heavily (since it scored higher),
    but include EI for its better-calibrated exploration in high-uncertainty
    regions where UCB might be overly aggressive.
    """
    start_time = time.time()
    deadline = start_time + 290

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # UCB component: (f_best - mu) + beta * sigma
    beta = 2.0
    ucb = (f_best - mu) + beta * sigma

    # EI component: (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Blend UCB and EI with alpha weighting
    alpha = 0.6
    combined = alpha * ucb + (1.0 - alpha) * ei

    # Cost-aware score
    scores = combined / cost

    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, -np.inf)

    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
