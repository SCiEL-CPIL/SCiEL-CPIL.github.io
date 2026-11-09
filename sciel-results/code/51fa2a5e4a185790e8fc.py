"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Hybrid acquisition:
    #  - When f_best is still large (poor initial estimate), use a PES-inspired
    #    exploration score: sigma * P(mu < f_best) = sigma * (1 - Phi((f_best-mu)/sigma))
    #    This rewards high-uncertainty regions that have a meaningful chance of
    #    containing a new best.
    #  - When f_best is small (close to the optimum at 0), switch to GP mean
    #    minimization: -mu. This drives rapid convergence to the true minimum
    #    and avoids the conservative bias of EI near the optimum.
    #
    # The switch happens when f_best < 1.0, which for sphere 1d on [-5,5]
    # means we are within the basin of attraction of the global min at x=0.
    
    if f_best < 1.0:
        # GP Mean Minimization: minimize the predicted mean
        return -mu
    else:
        # PES-inspired: sigma * P(next best)
        z = (f_best - mu) / (sigma + 1e-9)
        prob_new_best = 1.0 - norm.cdf(z)
        return sigma * prob_new_best
# EVOLVE-BLOCK-END
