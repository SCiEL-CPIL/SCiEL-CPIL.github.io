import numpy as np
from scipy.special import erf, exp

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    For minimization, the Expected Improvement is defined as:
    EI(x) = E[max(0, f_best - f(x))]
    
    We compute the EI for each candidate and divide by the square root of the cost
    to penalize expensive evaluations less severely than linear cost, allowing
    exploration of potentially better but slightly more expensive regions.
    Score(x) = EI(x) / sqrt(cost(x))
    
    Higher score indicates a more promising point to evaluate next.
    """
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sigma
    
    # Compute CDF and PDF of the standard normal distribution using scipy.special
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    
    # phi(z) = exp(-0.5 * z^2) / sqrt(2 * pi)
    pdf = exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Numerical stability: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: Use square root of cost to soften the penalty
    # This reduces the bias towards the cheapest region (near 0) and allows
    # the optimizer to consider higher-cost regions if the expected improvement is sufficient.
    scores = ei / np.sqrt(cost)
    
    return scores
