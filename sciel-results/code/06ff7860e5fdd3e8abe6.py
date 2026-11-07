import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    For minimization, the Expected Improvement is defined as:
    EI(x) = E[max(0, f_best - f(x))]
    
    We compute the EI for each candidate and divide by the cost to penalize
    expensive evaluations.
    Score(x) = EI(x) / cost(x)
    
    Higher score indicates a more promising point to evaluate next.
    """
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sigma
    
    # CDF and PDF of the standard normal distribution
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Numerical stability: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    scores = ei / cost
    
    return scores
