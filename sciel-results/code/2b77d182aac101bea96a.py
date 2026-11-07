"""Cost-aware Knowledge Gradient acquisition for Bayesian optimization on Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Computes the Knowledge Gradient (KG) acquisition score.
    
    KG is the expected improvement in the best observed value f_best
    after evaluating a candidate x. For minimization:
    KG(x) = E[max(f_best - f(x), 0)]
    
    Given f(x) ~ N(mu, sigma^2), this is:
    KG = (f_best - mu) * Phi(d) + sigma * phi(d)
    where d = (f_best - mu) / sigma.
    
    The score is normalized by the evaluation cost.
    """
    # Ensure sigma is positive for division
    s = np.maximum(sigma, 1e-9)
    
    # Calculate d = (f_best - mu) / sigma
    # f_best is a scalar, mu and s are arrays
    d = (f_best - mu) / s
    
    # Calculate PDF and CDF
    # Use scipy.stats.norm for numerical stability
    pdf = norm.pdf(d)
    cdf = norm.cdf(d)
    
    # KG formula: sigma * (d * Phi(d) + phi(d))
    # Note: if d is very large positive, cdf->1, pdf->0, term -> sigma*d = f_best-mu
    # if d is very large negative, cdf->0, pdf->0, term -> 0
    kg = s * (d * cdf + pdf)
    
    # Ensure non-negative (numerical errors might cause tiny negatives)
    kg = np.maximum(kg, 0.0)
    
    # Handle non-finite values
    kg = np.where(np.isfinite(kg), kg, 0.0)
    
    # Normalize by cost
    c = np.maximum(cost, 1e-9)
    scores = kg / c
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
