"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# We use a time guard to ensure we finish within the 300s limit per evaluation.
# The acquisition function is called once per iteration of the main loop (up to ~25 iterations).
# Each call processes 2000 candidates.
# We don't have a global timer across calls easily unless we use a module-level variable,
# but the task says "Each evaluation must finish within 300 s". This usually refers to the
# entire run or the specific function call?
# "Each evaluation" in the context of the contract "The evaluator measures the raw value... 
# Each evaluation must finish within 300 s" likely means the whole BO loop run for one instance.
# However, the function `acquisition` is called repeatedly.
# To be safe, we keep the acquisition function very fast.
# 2000 points is small. Standard EI is O(N).

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: (2000,) array of GP means
    sigma: (2000,) array of GP standard deviations
    f_best: scalar, the best (lowest) objective value observed so far
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of scores (higher is better)
    """
    # Ensure inputs are numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases for sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, we look for f(x) < f_best.
    # The improvement is f_best - f(x).
    # E[max(f_best - f(x), 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Cost-aware adjustment
    # We want to balance improvement and cost.
    # A common approach is to divide by cost, or use a weighted combination.
    # Given the cost model 1 + mean(x_scaled), costs are in [1, 2].
    # Dividing by cost is a simple and effective way to prefer cheaper points 
    # when the expected improvement is similar.
    
    # Alternative: UCB with cost penalty
    # UCB = mu - beta * sigma
    # Cost-aware UCB = UCB / cost
    
    # Let's stick to EI / cost as it is robust.
    # To avoid division by zero (though cost >= 1), we use safe cost.
    cost_safe = np.maximum(cost, 1e-12)
    
    scores = ei / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
