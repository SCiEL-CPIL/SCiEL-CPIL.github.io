"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 330 - 5  # Safety margin: kill program 5s before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Instead of maximizing Expected Improvement (EI), we sample from the 
    posterior distribution of the objective function at each candidate point 
    and divide by the cost. This encourages exploration of high-uncertainty 
    regions (where sigma is high) relative to their cost, helping to escape 
    local minima in multimodal functions like Ackley and Rastrigin.
    """
    # Deadline guard: if we are approaching the limit, return a safe fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: return cost-aware EI as a safe deterministic result
        z = (f_best - mu) / (sigma + 1e-9)
        from scipy.stats import norm
        return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost

    # Thompson Sampling: sample from N(mu, sigma)
    # Use a fixed random state? No, we want different samples each call to encourage exploration.
    # However, since this function is called once per loop iteration with 2000 candidates,
    # we can just draw one sample per candidate.
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample from the posterior
    # We want to minimize the objective, so we sample f(x) ~ N(mu, sigma)
    # Then we score based on f(x) / cost
    # To make it more robust, we can add a small exploration bonus or just rely on the sampling.
    
    # Draw samples
    # Note: We don't have access to the random seed of the evaluator, so we use np.random
    # This introduces stochasticity into the selection, which is the point of TS.
    samples = mu + sigma_safe * np.random.randn(len(mu))
    
    # Score: lower is better for the objective, but we need to return higher scores for "evaluate next"
    # So we invert the sample value and divide by cost.
    # score = -sample / cost
    # This means we prefer low f(x) and low cost.
    
    # Add a small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = -samples / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
# EVOLVE-BLOCK-END
