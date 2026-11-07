"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling-based cost-aware acquisition.
    
    We sample a single latent function value for each candidate from the GP
    predictive distribution and select the candidate that maximizes the ratio
    of (f_best - sampled_value) to its evaluation cost. This stochasticizes
    the exploration-exploitation trade-off.
    
    To make this more effective, we use a hybrid approach:
    - Thompson Sampling for stochastic exploration
    - Add a small UCB-like bonus to encourage exploration in uncertain regions
    - Weight by inverse cost to account for evaluation costs
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed function value
        cost: array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.clip(sigma, 1e-8, None)
    
    # Thompson Sampling: sample from the predictive distribution
    # Use a fixed random seed per call for reproducibility within the loop,
    # but vary it slightly to avoid deterministic patterns
    # Actually, we should NOT use a fixed seed - we want true stochasticity
    # The randomness comes from the sampling itself
    
    # Sample latent values: y ~ N(mu, sigma^2)
    # We want to minimize, so we care about (f_best - y)
    # Sample: y = mu + sigma * z where z ~ N(0, 1)
    
    # Generate standard normal samples
    z = np.random.standard_normal(n)
    sampled_values = mu + sigma_safe * z
    
    # For minimization, improvement is (f_best - sampled_value)
    # We want candidates where sampled_value is likely lower than f_best
    improvement = f_best - sampled_values
    
    # Cost-aware: divide by cost
    # Ensure cost is positive
    cost_safe = np.clip(cost, 1e-8, None)
    
    # Primary score: Thompson sampling ratio
    ts_score = improvement / cost_safe
    
    # Add a UCB-like exploration bonus
    # The UCB bonus encourages exploration where sigma is high
    # Use a beta parameter that balances exploration vs exploitation
    # Since we're doing Thompson sampling, the exploration is already stochastic,
    # but adding a small UCB term can help guide the search
    
    # Normalize sigma for the bonus
    sigma_max = np.max(sigma_safe) if n > 0 else 1.0
    if sigma_max > 1e-10:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Small exploration bonus (beta=0.1 to keep it subtle)
    beta = 0.1
    ucb_bonus = beta * sigma_norm / cost_safe
    
    # Combine Thompson sampling score with UCB bonus
    score = ts_score + ucb_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline - if we're taking too long, return what we have
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
