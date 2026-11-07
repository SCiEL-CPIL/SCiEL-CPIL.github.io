"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Thompson Sampling: draw a single sample from the posterior
    # and score by negative of the sample (lower is better, so we negate)
    rng = np.random.default_rng()
    z = rng.standard_normal(len(mu))
    theta = mu + sigma * z
    return -theta
# EVOLVE-BLOCK-END
