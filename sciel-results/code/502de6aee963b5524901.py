import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function for Bayesian Optimization.
    
    Instead of using Expected Improvement (EI) or UCB, this method calculates the 
    probability that a candidate point x will yield a value f(x) lower than the 
    current best f_best.
    
    For minimization, assuming f(x) ~ N(mu, sigma^2):
    PI(x) = P(f(x) < f_best) = P((f(x) - mu) / sigma < (f_best - mu) / sigma)
    PI(x) = norm.cdf((f_best - mu) / sigma)
    
    This approach favors points with a high likelihood of surpassing the current best,
    regardless of the magnitude of the potential improvement. It is distinct from EI 
    (which considers the size of improvement) and UCB (which uses a linear combination 
    of mean and uncertainty).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability: avoid division by zero or very small sigma
    # If sigma is extremely small, the posterior is effectively deterministic.
    # norm.cdf(z) -> 1 if z -> +inf, 0 if z -> -inf, 0.5 if z -> 0.
    # We clip sigma to a small positive value to prevent numerical issues in division.
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Probability of Improvement
    pi_scores = norm.cdf(z)
    
    # Ensure finite scores
    pi_scores = np.where(np.isfinite(pi_scores), pi_scores, 0.0)
    
    # The evaluator picks the candidate with the highest score.
    return pi_scores
