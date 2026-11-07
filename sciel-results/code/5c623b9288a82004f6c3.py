"""Cost-aware Knowledge Gradient acquisition for Bayesian optimization on Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def _kg_pair(mu1, s1, mu2, s2):
    """E[max(X1, X2) - X1] with X1 ~ N(mu1, s1^2), X2 ~ N(mu2, s2^2)."""
    sd = np.sqrt(s1 * s1 + s2 * s2)
    sd = np.maximum(sd, 1e-9)
    d = (mu2 - mu1) / sd
    pdf = norm.pdf(d)
    cdf = norm.cdf(d)
    # sd * (phi(d) - d * Phi(d))  ==  sd * (phi(d) + d * (1 - Phi(d)))
    return sd * (pdf + d * (1.0 - cdf))


def acquisition(mu, sigma, f_best, cost):
    # Improved Knowledge Gradient:
    # Uses the minimum of the current best observed value and the GP mean minimum
    # as the reference for expected improvement. This balances exploitation of
    # known good values with exploration of regions where the GP predicts low values.
    s = np.maximum(sigma, 1e-9)
    
    # Determine the reference value for improvement.
    # mu_min represents the most promising value according to the GP mean.
    mu_min = np.min(mu)
    # If the GP predicts a value lower than the best observed, use that as the target.
    # Otherwise, use the best observed value.
    ref = np.minimum(f_best, mu_min)
    
    # Calculate Expected Improvement relative to 'ref'.
    # This is E[max(ref - f(x), 0)].
    g = _kg_pair(mu, s, ref, 0.0)
    
    # Numerical safety: ensure finite, non-negative scores.
    g = np.where(np.isfinite(g), g, 0.0)
    
    # Normalize by cost
    c = np.maximum(cost, 1e-9)
    return g / c
# EVOLVE-BLOCK-END
