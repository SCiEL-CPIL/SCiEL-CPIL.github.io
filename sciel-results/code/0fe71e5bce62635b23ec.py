"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound: exploit low mu, explore high sigma
    # beta controls exploration-exploitation trade-off
    beta = 1.0
    # LCB: lower is better for the objective, so we want to minimize (mu - beta * sigma)
    # But we return scores where higher = evaluate next
    # So we return negative of LCB: -(mu - beta * sigma) = -mu + beta * sigma
    scores = -mu + beta * sigma
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    return scores
# EVOLVE-BLOCK-END
