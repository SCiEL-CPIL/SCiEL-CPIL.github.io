"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Probability of Improvement per unit cost.
    # POI is smoother than EI and often more robust in early stages.
    z = (f_best - mu) / (sigma + 1e-9)
    poi = norm.cdf(z)
    # Avoid division by zero, though cost is >= 1.5
    scores = poi / (cost + 1e-9)
    return scores
# EVOLVE-BLOCK-END
