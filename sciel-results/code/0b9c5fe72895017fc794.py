import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Compute Expected Improvement for minimization
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware scoring: stronger preference for cheap points
    # Use cost^1.5 for more aggressive cost discounting than linear
    cost_factor = cost ** 1.5
    
    # Add a small exploration bonus to avoid premature convergence
    # Normalize sigma by its mean to make the bonus scale-invariant
    sigma_mean = np.mean(sigma) + 1e-9
    exploration_bonus = 0.01 * sigma / sigma_mean
    
    # Final score: higher is better
    score = ei / cost_factor + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
