"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We need to be careful with the time limit. The evaluator kills after 330s.
# The function `acquisition` is called repeatedly. We don't know the exact call count,
# but the loop runs while total cost < 30.
# Initial cost: 5 points. Cost per point is 1 + mean(x_scaled).
# x_scaled is in [0, 1], so mean is in [0, 1]. Cost is in [1, 2].
# Average cost is roughly 1.5.
# 30 / 1.5 = 20 evaluations on average. Maybe up to 30 if points are cheap.
# So `acquisition` might be called ~20-30 times per instance.
# 5 instances * 30 calls = 150 calls total?
# Or is it 5 instances run sequentially? "mean over the five instances".
# Usually these tasks run the whole loop for each instance separately or in parallel?
# "Each function runs on five seeded instances".
# The evaluator likely calls `acquisition` for one instance, completes it, then moves to the next?
# Or does it manage the loop externally?
# "Write acquisition(mu, sigma, f_best, cost) for a loop... The evaluator scores 2000 new uniform candidates... and evaluates the best one."
# This implies the evaluator manages the loop.
# The evaluator calls `acquisition` once per iteration of the loop.
# The loop continues while total cost < 30.
# So `acquisition` is called multiple times.
# We must ensure each call is fast.
# 2000 points is small. Vectorized operations are fast.

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores.
    
    Parameters:
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviations
    f_best: float, the best function value observed so far
    cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
    (2000,) array of acquisition scores (higher is better)
    """
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Note: For minimization, EI is defined as E[max(f_best - f(x), 0)]
    
    z = (f_best - mu) / sigma
    
    # Use numerical stability tricks for large z
    # norm.cdf(z) is 1 for large z, 0 for small z
    # norm.pdf(z) is small for large |z|
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Cost-aware: divide by cost
    # cost is in [1, 2]
    scores = ei / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
