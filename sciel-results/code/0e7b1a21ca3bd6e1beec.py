import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling with Cost Penalty.
    
    Strategy:
    1. Sample function values from the posterior predictive distribution:
       f_sample(x) ~ N(mu(x), sigma(x)^2)
    2. Select the point that minimizes the ratio f_sample(x) / cost(x).
       This balances finding low function values (exploitation) with
       low evaluation costs (efficiency), while the stochastic sampling
       provides exploration.
    
    We use a single random seed for the acquisition call to ensure
    reproducibility within a single evaluation step, but different
    calls will explore different parts of the space.
    """
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-6)
    
    # Generate a random sample from the posterior predictive distribution
    # We use a fixed seed per call to make the function deterministic given the inputs,
    # but the randomness comes from the fact that we are simulating a draw.
    # Actually, Thompson Sampling requires randomness. The function signature doesn't allow
    # passing a random state, so we must use the global numpy random state.
    
    # To avoid issues with the global state being shared or reset, we can use a local
    # RandomGenerator seeded with a hash of the inputs or just use np.random.
    # Using np.random is standard for these tasks unless specified otherwise.
    
    # Draw a random number z ~ N(0, 1) for each candidate
    z = np.random.randn(len(mu))
    
    # Sampled function values
    f_sample = mu + sigma * z
    
    # Compute the score: we want to maximize the "value" of the evaluation.
    # In minimization, lower f_sample is better.
    # Cost-aware: we want low f_sample and low cost.
    # A common approach is to minimize f_sample / cost, which is equivalent to maximizing cost / f_sample (if f_sample > 0) or similar.
    # However, f_sample can be negative.
    # A robust cost-aware score for minimization is: Score = -f_sample / cost.
    # Maximizing this is equivalent to minimizing f_sample / cost.
    # This prioritizes points that are both predicted to have low function values and low costs.
    
    # Alternative: Use a Lagrangian-like penalty.
    # Score = -f_sample - lambda * cost. But lambda is hard to set.
    # The ratio approach is more adaptive to the scale of f_sample.
    
    # Let's use Score = -f_sample / cost.
    # If f_sample is very negative (good), Score is large positive.
    # If f_sample is positive (bad), Score is negative.
    # If cost is high, the magnitude of the score is reduced.
    
    scores = -f_sample / cost
    
    # Handle non-finite values
    scores[~np.isfinite(scores)] = -1e9
    
    return scores
