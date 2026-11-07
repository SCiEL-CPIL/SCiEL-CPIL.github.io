"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Lower Confidence Bound (LCB) for minimization, normalized by cost.
    # Maximizes (beta * sigma - mu) / cost.
    # beta = 1.0 provides a balance between exploration (sigma) and exploitation (-mu).
    beta = 1.0
    score = (beta * sigma - mu) / cost
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    return score
# EVOLVE-BLOCK-END
