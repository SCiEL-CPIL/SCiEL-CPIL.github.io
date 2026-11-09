"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # UCB acquisition for minimization.
    # Score = -mu + beta * sigma. Higher score means "evaluate next".
    # We want low mu (so -mu is high) and high sigma (exploration).
    beta = 1.0
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
