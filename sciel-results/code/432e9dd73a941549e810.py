"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = evaluate next) for the 2000 candidates.
    
    Uses a structurally different approach: Thompson Sampling with cost-aware
    weighting. Instead of deterministic EI, we sample from the posterior at each
    candidate and score by the improvement potential divided by cost. This
    naturally balances exploration and exploitation and handles the cost
    constraint through stochastic sampling rather than a fixed penalty.
    
    Additionally, we add a small UCB component for robustness.
    """
    start_time = time.time()
    DEADLINE = 320.0  # seconds, safety margin before 330s limit
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Thompson Sampling: sample one value from N(mu, sigma^2) for each candidate
    rng = np.random.default_rng()
    samples = mu + sigma_safe * rng.standard_normal(n)
    
    # Improvement potential: how much better than f_best could this be?
    # We use the negative of the sample relative to f_best, clipped at 0
    # This represents the expected improvement if we were to evaluate here
    improvement = np.maximum(f_best - samples, 0.0)
    
    # Cost-aware score: improvement divided by cost
    # This prioritizes candidates that offer good improvement per unit cost
    score = improvement / cost_safe
    
    # Add a small UCB component for exploration in high-uncertainty regions
    # The UCB term encourages exploring areas where we're uncertain
    # We use a temperature that adapts: higher when we have fewer observations
    # Since we don't know the exact iteration count, we use a fixed small bonus
    ucb_bonus = 0.05 * sigma_safe / cost_safe
    
    # Combine Thompson Sampling score with UCB exploration bonus
    score = score + ucb_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # If we're close to the deadline, return quickly
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
