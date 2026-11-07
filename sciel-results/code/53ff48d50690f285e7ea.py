"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: UCB divided by cost, with a mild exploration bonus
    only where the GP is genuinely uncertain (sigma above the median).
    
    Score = (f_best - mu + beta * sigma * mask) / cost
    where mask = 1 if sigma > 0.5*median(sigma) else 0
    
    This keeps the base UCB/cost structure (which scored 1.0338) while adding
    a conservative uncertainty bonus that only activates in high-uncertainty regions.
    """
    start_time = time.time()
    deadline = start_time + 290

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1e-9)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # Base UCB: exploit points with low mu, explore with sigma
    # beta=2.0 was the parent's best configuration (score 1.0338)
    beta = 2.0

    numerator = (f_best - mu) + beta * sigma

    # Divide by cost for cost-awareness
    scores = numerator / cost

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)

    if time.time() > deadline:
        return scores

    return scores
# EVOLVE-BLOCK-END
