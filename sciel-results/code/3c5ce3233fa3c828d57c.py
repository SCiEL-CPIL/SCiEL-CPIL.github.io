"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # Assuming f(x) ~ N(mu, sigma^2)
    
    # Ensure sigma is positive to avoid division by zero issues in z-score
    # If sigma is 0, the function value is deterministic.
    # We add a tiny epsilon to sigma to prevent numerical issues, 
    # but we handle the limit case logically if needed. 
    # Actually, for very small sigma, z -> +/- inf.
    # norm.cdf(inf) = 1, norm.cdf(-inf) = 0.
    # norm.pdf(inf) = 0.
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: Phi(z) is the probability that f(x) < f_best is NOT what we want directly in that form.
    # Let Y = f_best - f(x). Y ~ N(f_best - mu, sigma^2).
    # We want E[max(Y, 0)].
    # E[max(Y, 0)] = (f_best - mu) * P(Y > 0) + sigma * phi(z)
    # P(Y > 0) = P(f(x) < f_best) = Phi((f_best - mu) / sigma) = Phi(z)
    
    # Standard formula for minimization EI:
    # EI = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative values due to numerical precision (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
