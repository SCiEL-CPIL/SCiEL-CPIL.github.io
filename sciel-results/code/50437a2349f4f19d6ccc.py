"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB): -mu + kappa * sigma
    # Higher score means more promising for next evaluation
    # Adaptive kappa: explore more when f_best is high, exploit more when low
    if f_best > 3.0:
        kappa = 2.5
    elif f_best > 1.0:
        kappa = 1.8
    else:
        kappa = 1.2
    
    score = -mu + kappa * sigma
    return score
# EVOLVE-BLOCK-END
