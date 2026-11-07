"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We need to be careful about time. The evaluator kills after 330s.
# The acquisition function is called in a loop.
# We don't have access to the global loop state directly, but we can infer it or just be fast.
# However, the prompt says "Each evaluation must finish within 300 s". This likely refers to the whole run or each step?
# "The evaluator measures... The search score is seed/value... Keep the function names... Each evaluation must finish within 300 s."
# Usually "evaluation" in these contexts might mean the entire optimization run per instance, or the step.
# Given "While total cost is below 30... evaluates the best one", the loop runs until cost >= 30.
# Initial cost: 5 points. Cost of point x is 1 + mean(x_scaled).
# x_scaled is in [0,1]. Mean is in [0,1]. So cost per point is in [1, 2].
# Max evaluations: 30 / 1 = 30 points. Min: 30 / 2 = 15 points.
# So the loop runs ~15-25 times.
# Each call to acquisition processes 2000 candidates.
# Total time budget 300s per "evaluation" (likely per instance run).
# 20-25 calls * 2000 points.
# Standard EI is O(N).
# We can add a time guard if needed, but EI is very fast.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    mu : np.ndarray of shape (2000,) - Mean of GP
    sigma : np.ndarray of shape (2000,) - Std dev of GP
    f_best : float - Best observed value
    cost : np.ndarray of shape (2000,) - Cost of evaluating each point
    
    Returns:
    scores : np.ndarray of shape (2000,) - Acquisition scores (higher is better)
    """
    # Handle potential numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard Expected Improvement (EI)
    # Improvement is defined as f_best - mu (since we are minimizing)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF, phi is PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost-aware: EI / cost
    # Cost is 1 + mean(x_scaled). x_scaled is in [0,1].
    # Lower cost is better, so we divide by cost.
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
