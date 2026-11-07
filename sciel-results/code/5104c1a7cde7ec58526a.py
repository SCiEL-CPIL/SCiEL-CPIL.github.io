"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # UCB with beta=1.0 for a balance of exploration and exploitation
    beta = 1.0
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
