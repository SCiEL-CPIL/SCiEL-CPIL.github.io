# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function based on stochastic knapsack principle.
    
    Maximizes: P(mu < f_best) * (f_best - mu) / cost
    
    This metric weights the potential improvement (f_best - mu) by the probability
    that the true value is actually better than the current best, and normalizes
    by the evaluation cost.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle sigma to avoid division by zero
    # Use a very small epsilon to prevent numerical issues
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate the standardized distance z
    # z = (f_best - mu) / sigma
    # If mu < f_best, z > 0. If mu > f_best, z < 0.
    z = (f_best - mu) / sigma_safe
    
    # Calculate P(mu < f_best) which is Phi(z)
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # Using erf directly can be slightly faster and avoids importing scipy.stats
    # which might have overhead, though negligible here.
    # We clip z to avoid unnecessary computation for extreme values where erf is ~ +/- 1
    # erf(x) for x > ~8 is 1.0, for x < ~-8 is -1.0.
    # However, erf handles large inputs gracefully.
    
    # Compute CDF
    # Note: erf(z / sqrt(2))
    sqrt2 = np.sqrt(2.0)
    prob_improve = 0.5 * (1.0 + erf(z / sqrt2))
    
    # Calculate the improvement magnitude
    # This is positive if mu < f_best, negative if mu > f_best
    improvement = f_best - mu
    
    # Probability-weighted improvement
    # If mu > f_best, improvement is negative, prob_improve is small positive.
    # Product is negative.
    # If mu < f_best, improvement is positive, prob_improve is large positive.
    # Product is positive.
    weighted_improvement = prob_improve * improvement
    
    # Cost-aware normalization
    cost_safe = np.maximum(cost, eps)
    scores = weighted_improvement / cost_safe
    
    # Ensure all scores are finite
    # Replace any non-finite values (inf, nan) with 0.0
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
