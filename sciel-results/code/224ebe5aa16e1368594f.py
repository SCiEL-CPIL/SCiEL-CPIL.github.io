"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB)
    # We want to maximize the score, so we return -mu + beta * sigma
    # Beta is a hyperparameter controlling exploration.
    # A value of 1.5 provides a balance between exploitation and exploration.
    beta = 1.5
    scores = -mu + beta * sigma
    return scores
# EVOLVE-BLOCK-END
