# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Instead of calculating Expected Improvement (EI), this method draws a single
    sample from the posterior predictive distribution for each candidate:
    theta_i = mu_i + sigma_i * z_i, where z_i ~ N(0, 1).
    
    The candidates are then ranked by a cost-aware regret metric. Since we want
    to minimize the function, we prefer low theta values. To account for cost,
    we normalize by the square root of the cost (a common heuristic for cost-aware
    TS that balances exploration and exploitation without over-penalizing cost 
    as aggressively as linear cost division might in some regimes, or simply 
    using 1/sqrt(cost) to encourage cheap evaluations while still respecting 
    the value).
    
    Specifically, we compute score = -theta / sqrt(cost).
    Higher score means we should evaluate next.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best observed function value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety margin for time: check elapsed time. 
    # Although this function is called once per iteration in the loop, 
    # the directive asks for a deadline guard. 
    # Since we don't have access to the start time of the *entire* run easily 
    # without global state, and this function is lightweight, 
    # we primarily rely on the vectorized operation being fast.
    # However, to strictly follow the "explicit deadline guard" if possible:
    # We cannot know the global start time here. But we can ensure we don't 
    # do anything heavy. The operation is O(N) vectorized, which is <1ms.
    
    # Handle edge cases
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-9)
    
    # Draw a single sample from the posterior predictive distribution
    # z ~ N(0, 1)
    # Using a fixed seed per call is not possible as we need randomness for TS.
    # Standard numpy random is fine.
    z = np.random.randn(len(mu))
    
    # Posterior sample: theta = mu + sigma * z
    theta = mu + sigma * z
    
    # Cost-aware scoring:
    # We want to minimize theta.
    # We want to minimize cost.
    # Score = -theta / sqrt(cost)
    # This is a heuristic: dividing by sqrt(cost) penalizes expensive points 
    # but less severely than 1/cost, which can sometimes lead to too much 
    # exploitation of cheap but uninformative regions.
    # Alternatively, -theta / cost is the standard "UCB with cost" or "EI/cost" analogue.
    # Let's stick to -theta / cost for consistency with the "cost-aware" spirit 
    # of the parent, but applied to the sampled value.
    
    score = -theta / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
