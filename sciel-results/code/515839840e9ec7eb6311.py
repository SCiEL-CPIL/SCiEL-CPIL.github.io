"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a discrete, cost-normalized quantile-based heuristic.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, the best observed value so far.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of scores. Higher is better.
    """
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Prevent division by zero in sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate Z-score for P(f(x) < f_best)
    # Since we are minimizing, improvement is when f(x) is lower than f_best.
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Discretize the probability of improvement into quantile bins
    # This creates a discrete ranking that is less sensitive to small changes in mu/sigma
    # Use 100 quantile bins for a fine-grained but discrete representation
    n_bins = 100
    # Clip pi to [0, 1] for safety
    pi_clipped = np.clip(pi, 0.0, 1.0)
    
    # Create discrete quantile ranks
    # For each candidate, determine which quantile bin its pi falls into
    # We use the empirical quantile of the pi distribution to create uniform bins
    # This ensures that the ranking is based on relative improvement likelihood
    # among the current candidates, not absolute probability values
    
    # Create bin edges based on uniform quantiles of the pi distribution
    # Use np.quantile to get the bin edges
    # Add a small epsilon to avoid edge cases
    pi_for_quantiles = pi_clipped.copy()
    # Ensure we have enough unique values for quantile calculation
    if np.unique(pi_for_quantiles).size < n_bins:
        # If not enough unique values, add small noise to break ties
        pi_for_quantiles = pi_for_quantiles + np.random.uniform(0, 1e-12, size=pi_for_quantiles.shape)
    
    # Calculate quantile edges
    quantile_edges = np.quantile(pi_for_quantiles, np.linspace(0, 1, n_bins + 1))
    # Ensure edges are strictly increasing
    for i in range(1, len(quantile_edges)):
        if quantile_edges[i] <= quantile_edges[i-1]:
            quantile_edges[i] = quantile_edges[i-1] + 1e-12
    
    # Assign each candidate to a bin
    # Use np.searchsorted to find the bin index
    bin_indices = np.searchsorted(quantile_edges, pi_clipped, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Convert bin indices to a normalized score in [0, 1]
    # Higher bin index means higher probability of improvement
    quantile_score = (bin_indices + 1) / n_bins
    
    # Prevent division by zero in cost
    cost_safe = np.maximum(cost, 1e-10)
    
    # Score: Ratio of quantile-based improvement score to cost
    # This prioritizes candidates with high relative improvement likelihood and low cost
    scores = quantile_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
