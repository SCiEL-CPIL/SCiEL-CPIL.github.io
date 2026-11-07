"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Samples from the GP posterior distribution for each candidate and
    selects candidates based on the ratio of sampled value to cost.
    This stochastic approach balances exploration and exploitation differently
    than deterministic EI-based methods.
    
    Parameters:
    mu (np.ndarray): GP posterior means, shape (N,)
    sigma (np.ndarray): GP posterior standard deviations, shape (N,)
    f_best (float): Best observed function value so far.
    cost (np.ndarray): Evaluation costs for each candidate, shape (N,)
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher scores indicate
                candidates that are more likely to be evaluated next.
    """
    # Deadline guard: track elapsed time
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit with 20s safety margin
    
    # Convert to numpy arrays for consistent typing
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    N = len(mu)
    
    # Ensure sigma is non-zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: sample from GP predictive distribution
    # f(x) ~ N(mu, sigma^2)
    # We want to minimize f(x), so we sample and prefer lower values
    # But the acquisition score should be higher = better to evaluate
    
    # For minimization with Thompson Sampling:
    # Sample y ~ N(mu, sigma^2)
    # Score should reward points that likely have low f(x) relative to cost
    # We can use: score = -y / cost (higher is better, since lower y is good)
    # Or equivalently: score = (f_best - y) / cost + exploration_bonus
    
    # Use a hybrid approach:
    # 1. Sample from the GP posterior
    # 2. Compute a score that balances improvement over f_best and cost
    
    rng = np.random.default_rng()
    
    # Sample latent function values
    samples = mu + sigma_safe * rng.standard_normal(N)
    
    # Compute improvement over f_best (positive if improvement)
    improvement = f_best - samples
    
    # Cost-aware score: improvement divided by cost
    # Add a small exploration bonus based on uncertainty to encourage exploring
    # uncertain regions, but keep it modest to avoid pure exploration
    
    # Base score: expected improvement per unit cost
    scores = improvement / cost
    
    # Add exploration bonus proportional to sigma (uncertainty)
    # This encourages exploring uncertain regions
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe / cost
    
    scores = scores + exploration_bonus
    
    # Replace any non-finite values with 0.0 to ensure validity
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
