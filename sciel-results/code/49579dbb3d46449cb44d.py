import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Instead of deterministic Expected Improvement (EI) or Probability of Improvement (PI),
    we draw a single sample from the GP posterior for all 2000 candidates.
    The score is the negative sampled function value divided by the evaluation cost.
    This promotes exploration (high sigma leads to larger variance in samples) 
    and exploitation (low mu leads to lower sampled values) while respecting the cost constraint.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 285:
        # Fallback to a simple, fast deterministic metric if time is running out
        # Use a robust UCB-like score divided by cost
        z = (f_best - mu) / (sigma + 1e-9)
        # Simple fallback: maximize (f_best - mu) / cost
        # This is essentially a greedy improvement heuristic
        return (f_best - mu) / (cost + 1e-9)

    # Thompson Sampling:
    # 1. Draw a single sample from the posterior distribution for all 2000 points.
    #    y ~ N(mu, sigma)
    #    We use a fixed seed relative to the call count or just random.
    #    Since we don't have a global counter, we use np.random which is seeded by the environment.
    #    However, to be safe and deterministic within a run if possible, we could use a hash,
    #    but standard Thompson Sampling relies on randomness.
    #    The task says "five seeded instances", implying the environment might be seeded.
    #    We will use np.random.normal.
    
    # Ensure sigma is positive and finite
    sigma_safe = np.where(sigma > 1e-6, sigma, 1e-6)
    
    # Draw sample
    # Note: mu and sigma are in objective units.
    y_sample = mu + sigma_safe * np.random.randn(len(mu))
    
    # 2. Calculate the score.
    # We want to minimize the function value f(x).
    # The sampled value y_sample is an estimate of f(x).
    # We want to pick the x with the lowest expected cost-adjusted value.
    # Score = -y_sample / cost
    # Higher score is better.
    
    # Handle potential numerical issues in cost
    cost_safe = np.where(cost > 1e-6, cost, 1e-6)
    
    score = -y_sample / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
