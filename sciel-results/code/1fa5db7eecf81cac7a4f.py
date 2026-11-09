"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB (Upper Confidence Bound) for minimization.
    # We want to minimize, so we prefer low mu and high sigma (uncertainty).
    # Score = -mu + kappa * sigma
    # Higher score means better.
    kappa = 1.5
    scores = -mu + kappa * sigma
    return scores
# EVOLVE-BLOCK-END
