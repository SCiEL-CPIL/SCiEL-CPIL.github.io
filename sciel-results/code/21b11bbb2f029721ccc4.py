# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization.
    # This refines the Probability of Improvement (PI) design by incorporating
    # the expected magnitude of improvement, which helps avoid stalling on
    # regions with high probability of improvement but low expected gain.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard against degenerate / negative std (sigma is a standard deviation).
    s = np.clip(sigma, 1e-9, None)

    z = (f_best - mu) / s
    z = np.clip(z, -38.0, 38.0)  # keep CDF/PDF finite and well-behaved

    cdf = norm.cdf(z)
    pdf = norm.pdf(z)

    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    scores = (f_best - mu) * cdf + s * pdf
    
    # Ensure scores are finite and non-negative
    scores = np.maximum(scores, 0.0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
