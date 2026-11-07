"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Best-of-K Thompson Sampling.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs for each candidate
    
    Returns:
        (2000,) array of scores (higher = evaluate next)
    """
    # Use multiple samples to reduce variance and improve exploration/exploitation balance
    K = 20
    rng = np.random.default_rng()
    
    # Draw K samples from the posterior at each candidate point
    # samples shape: (K, 2000)
    # mu and sigma are broadcasted to (1, 2000)
    eps = rng.standard_normal(size=(K, mu.shape[0]))
    samples = mu[None, :] + sigma[None, :] * eps
    
    # For minimization, we want to find low values.
    # Score = -y_sample / cost (higher is better)
    # Add a small epsilon to avoid division by zero
    safe_cost = cost + 1e-9
    # scores shape: (K, 2000)
    scores = -samples / safe_cost[None, :]
    
    # Take the maximum score across the K samples for each candidate.
    # This effectively selects the candidate that has the best potential outcome
    # in the posterior samples, balancing exploration (chance of a good sample)
    # and exploitation (consistently good samples).
    final_scores = np.max(scores, axis=0)
    
    # Ensure all scores are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, -np.inf)
    
    return final_scores
# EVOLVE-BLOCK-END
