import numpy as np
import time

# Module-level state to track time for the deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: scalar, lowest observed value.
        cost: (2000,) array of costs for each candidate.
        
    Returns:
        (2000,) array of acquisition scores (higher is better).
    """
    global _start_time
    
    # Initialize start time if not set
    if _start_time is None:
        _start_time = time.time()

    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 295.0:
        # Fallback: pick the cheapest point with lowest mean
        # Score = -cost - 0.1 * mu (prioritize cost, then exploitation)
        mu_arr = np.asarray(mu, dtype=float)
        cost_arr = np.asarray(cost, dtype=float)
        mu_arr = np.nan_to_num(mu_arr, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_arr = np.nan_to_num(cost_arr, nan=1.0, posinf=1e6, neginf=-1e6)
        return -(cost_arr + 0.1 * mu_arr)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure finite values and valid ranges
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=-1e6)
    
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: Draw a deterministic posterior sample
    # Use a deterministic seed based on the current state to ensure reproducibility
    # but different samples each iteration. We can use a hash of the current mu/sigma
    # or simply a fixed seed offset by call count. Since we don't have call count here,
    # we can use a deterministic seed derived from the data itself to ensure
    # different samples for different GP states, but consistent for the same state.
    # However, standard TS usually uses a random seed. To be safe and deterministic
    # across runs for the same input, we use a fixed seed. But this might lead to
    # the same sample every time if the GP doesn't change enough.
    # A better approach for deterministic TS in a stateless function is to use
    # a pseudo-random number generator seeded by a hash of the input arrays.
    
    # Let's use a simple deterministic seed based on the sum of mu and sigma
    # to ensure we get a "random" sample that is deterministic for the same input.
    seed_val = int(np.sum(mu) * 1e6 + np.sum(sigma) * 1e6 + f_best * 1e6)
    # Ensure seed is non-negative and within valid range
    seed_val = abs(seed_val) % (2**32)
    
    rng = np.random.default_rng(seed_val)
    
    # Draw sample from N(mu, sigma^2)
    # y_sample ~ N(mu, sigma^2)
    y_sample = mu + sigma * rng.standard_normal(size=mu.shape)
    
    # We want to minimize y_sample / cost
    # So the score should be - (y_sample / cost)
    # Higher score means lower sampled cost-adjusted objective.
    
    score = - (y_sample / cost)
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=-1e6, posinf=-1e6, neginf=1e6)
    
    return score
