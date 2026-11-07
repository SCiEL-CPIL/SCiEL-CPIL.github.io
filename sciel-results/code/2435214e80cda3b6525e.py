# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Regret: P(f < f_best) = Phi((f_best - mu) / sigma).
    # A structurally different objective from EI: it rewards the *probability*
    # of finding a new best rather than the expected *magnitude* of improvement.
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against degenerate / negative std (sigma is a standard deviation).
    s = np.clip(sigma, 1e-9, None)

    z = (f_best - mu) / s
    z = np.clip(z, -38.0, 38.0)  # keep CDF finite and well-behaved

    # Higher score == more likely to be a new best == evaluate next.
    return norm.cdf(z)
# EVOLVE-BLOCK-END
