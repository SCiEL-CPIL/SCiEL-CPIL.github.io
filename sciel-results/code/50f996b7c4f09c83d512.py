"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Thinning-based Expected Improvement (EI) acquisition function.
    This approximates the Knowledge Gradient by subtracting the expected
    improvement of the top-k candidates from the primary candidate's EI.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Numerical stability
    epsilon = 1e-9
    sigma_safe = np.maximum(sigma, epsilon)
    cost_safe = np.maximum(cost, epsilon)
    
    # 1. Compute base Expected Improvement (EI) for minimization
    # We want to minimize, so improvement is f_best - f(x).
    # z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Phi is CDF, phi is PDF
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # 2. Thinning / Knowledge Gradient approximation
    # Subtract the expected improvement of the top-k candidates
    # This models the probability that the current point will remain
    # the best after a batch of evaluations.
    k = 5
    if n > k:
        # Find the indices of the top-k candidates by EI
        top_k_indices = np.argpartition(ei, -k)[-k:]
        top_k_ei = ei[top_k_indices]
        
        # For each candidate, subtract the sum of EI of the top-k candidates
        # that have higher EI than the current candidate
        # This is an approximation of the KG effect
        # A simpler approach: subtract the mean of top-k EI scaled by a factor
        # Or: for each point, subtract the number of top-k points that beat it
        # times their average EI
        
        # More precise thinning: for each point i, subtract sum of EI_j for j in top-k where EI_j > EI_i
        # This is O(n*k) which is fine for n=2000, k=5
        thinning_penalty = np.zeros(n)
        for j in top_k_indices:
            ei_j = ei[j]
            # Points with EI < ei_j get penalized by ei_j
            # Points with EI >= ei_j get no penalty from this j
            mask = ei < ei_j
            thinning_penalty[mask] += ei_j
        
        # Scale the penalty to be a fraction of the EI
        # The penalty should reduce the score of points that are likely to be
        # beaten by other high-EI points
        ei_thinned = ei - 0.5 * thinning_penalty
        ei_thinned = np.maximum(ei_thinned, 0.0)
    else:
        ei_thinned = ei
    
    # 3. Cost-aware penalty
    # Cost is 1 + mean(x_scaled). Min cost is 1.
    # Use a moderate power to balance exploration and cost
    cost_penalty = np.power(cost_safe, 0.5)
    
    scores = ei_thinned / cost_penalty
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
