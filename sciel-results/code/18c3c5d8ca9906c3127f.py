import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero in z calculation
    # sigma is a standard deviation, should be non-negative.
    # Add a small epsilon to prevent division by zero if sigma is exactly 0.
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If sigma is very small, z is large.
    # norm.cdf(large) -> 1, norm.pdf(large) -> 0.
    # EI -> f_best - mu.
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Structural Change: Quadratic Cost Penalty
    # The measured knowledge indicates that 1/cost^2 outperforms 1/cost.
    # This aggressively prioritizes low-cost evaluations to maximize the number
    # of samples within the fixed budget, improving the chance of finding
    # the global minimum.
    cost_penalty = cost ** 2
    
    # Calculate final score
    # Higher score means more attractive.
    # If EI is negative (mu > f_best), EI is 0? 
    # Actually, if mu > f_best, z is negative.
    # cdf(z) < 0.5, pdf(z) > 0.
    # (f_best - mu) is negative.
    # Term 1: negative * positive = negative.
    # Term 2: positive * positive = positive.
    # EI can be negative if the probability of improvement is low and the expected gain is small.
    # Standard EI is defined as max(0, ...).
    # Let's ensure EI is non-negative.
    ei = np.maximum(ei, 0.0)
    
    scores = ei / cost_penalty
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
