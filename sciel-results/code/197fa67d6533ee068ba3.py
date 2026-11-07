"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Probability of Improvement (PI) for minimization
    # P(Y < f_best) where Y ~ N(mu, sigma)
    # This encourages exploration by favoring points with high probability
    # of being better than the current best, regardless of the magnitude
    # of the expected improvement. This is often more robust for
    # multimodal functions like Weierstrass in short budgets.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score: (f_best - mu) / sigma
    # If mu is much lower than f_best, z is large positive, PI -> 1
    # If mu is close to f_best, z is near 0, PI -> 0.5
    # If mu is higher than f_best, z is negative, PI -> 0
    z = (f_best - mu) / sigma_safe
    
    # Compute Probability of Improvement
    # norm.cdf(z) gives P(Z < z) for standard normal
    # Since Y ~ N(mu, sigma), P(Y < f_best) = P((Y-mu)/sigma < (f_best-mu)/sigma) = Phi(z)
    pi_scores = norm.cdf(z)
    
    # Ensure finite values
    pi_scores = np.nan_to_num(pi_scores, nan=0.5, posinf=1.0, neginf=0.0)
    
    # Scale scores to be comparable to other metrics if needed, 
    # but relative ranking is what matters for the evaluator.
    # Higher PI is better.
    return pi_scores
# EVOLVE-BLOCK-END
