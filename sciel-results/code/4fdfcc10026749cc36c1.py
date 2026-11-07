"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian Optimization.
    
    Uses a hybrid approach combining:
    - Expected Improvement (EI) for exploitation
    - UCB component for exploration
    - Cost-aware normalization to prioritize cheaper evaluations
    
    The score is higher for candidates that are expected to improve f_best
    significantly relative to their evaluation cost.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate improvement over f_best
    improvement = f_best - mu
    
    # Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)]
    # For Gaussian f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = improvement / sigma_safe
    from scipy.stats import norm
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # UCB component for exploration
    # Beta controls exploration-exploitation tradeoff
    beta = 2.5
    ucb = -mu + beta * sigma_safe
    
    # Blend EI and UCB
    # EI is in objective units, UCB is also in objective units
    # Use a weighted combination
    weight_ei = 0.7
    weight_ucb = 0.3
    combined = weight_ei * ei + weight_ucb * ucb
    
    # Cost-aware scoring: divide by cost to get value per unit cost
    score = combined / cost_safe
    
    # Ensure all returned scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
