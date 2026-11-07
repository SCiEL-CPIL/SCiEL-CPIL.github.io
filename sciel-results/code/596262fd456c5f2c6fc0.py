import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Hypervolume Improvement (HVI) surrogate.
    
    For single-objective optimization, the expected hypervolume improvement over 
    the current best solution is mathematically equivalent to Expected Improvement (EI).
    We compute EI and apply a cost-based penalty to prefer cheaper evaluations.
    
    Score = EI(x) / cost(x)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero and numerical instability
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score: (f_best - mu) / sigma
    # This represents how many standard deviations below f_best the mean is.
    # Higher z means higher probability of improvement.
    z = (f_best - mu) / sigma_safe
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal distribution.
    
    # Clip z to prevent extreme values causing numerical issues in pdf, 
    # though norm.pdf handles large z well by returning 0.
    # norm.cdf handles large z well by returning 1.
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Calculate EI
    # If mu >= f_best, z is negative. EI should be small/zero.
    # The formula naturally handles this:
    # (f_best - mu) is negative, Phi(z) is small, sigma*phi(z) is positive but small.
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical artifacts might cause tiny negatives)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware weighting
    # We divide by cost to prioritize lower-cost points.
    # The cost is in [1, 2].
    # Using a power of 1.0 is a standard baseline.
    # If we want to be more aggressive about cost, we could use 1.5, 
    # but 1.0 is usually sufficient for linear cost models.
    scores = ei / (cost + eps)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
