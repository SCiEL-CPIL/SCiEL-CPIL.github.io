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
    # Knowledge Gradient: expected reduction in the best observed value
    # after evaluating the candidate, normalized by cost.
    s = np.maximum(sigma, 1e-9)
    g = _kg_pair(mu, s, f_best, 0.0)
    # Numerical safety: ensure finite, non-negative scores.
    g = np.where(np.isfinite(g), g, 0.0)
    c = np.maximum(cost, 1e-9)
    return g / c
# EVOLVE-BLOCK-END
