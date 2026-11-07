"""Acquisition function for Bayesian optimization of Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # UCB with beta=1 for more exploration
    beta = 1.0
    scores = mu + beta * sigma
    return scores
# EVOLVE-BLOCK-END
