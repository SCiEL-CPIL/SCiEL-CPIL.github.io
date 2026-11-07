import numpy as np
import time

# Module-level state for deadline tracking and caching
_start_time = time.time()
_DEADLINE = _start_time + 290.0  # 300s limit, 10s safety margin

# Cache to avoid redundant computation if called multiple times in same "batch"
# But since the contract says "Each function runs on five seeded instances",
# and the loop calls acquisition once per iteration, we need to be careful.
# However, the signature is acquisition(mu, sigma, f_best, cost).
# The problem states: "The evaluator scores 2000 new uniform candidates with acquisition and evaluates the best one."
# This implies acquisition is called once per iteration with arrays of shape (2000,).

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a safe, finite array. 
        # If we are about to timeout, we should return something that doesn't crash.
        # Returning zeros is safe.
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma and cost to avoid division by zero or negative values
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling Variant
    # Sample a single scalar offset from a Gaussian distribution.
    # Mean: f_best (current best observed value)
    # Std: Scales with max uncertainty. 
    # We use max(sigma) as a proxy for global uncertainty.
    # However, max(sigma) can be very large early on.
    # Let's scale it appropriately.
    # The objective is Ackley 2D on [-5, 5]^2. Range is roughly [0, 29].
    # Standard deviation of the GP posterior might be in that range.
    
    max_sigma = np.max(sigma)
    
    # Adaptive standard deviation for the offset
    # Early on, max_sigma is large. We want to explore.
    # Later, max_sigma is small. We want to exploit.
    # Using max_sigma directly as the std for the offset seems reasonable.
    # The offset represents a "virtual best" or a threshold.
    # We score candidates as -(mu - offset) / cost.
    # This is equivalent to maximizing (offset - mu) / cost.
    # If offset > mu, the score is positive (good).
    # If offset < mu, the score is negative (bad).
    # By sampling offset ~ N(f_best, max_sigma), we stochastically shift the target.
    
    # To prevent extreme values, we can clamp max_sigma if needed, but Ackley is bounded.
    # Let's use the sampled offset.
    offset = np.random.normal(loc=f_best, scale=max_sigma)
    
    # Score calculation
    # We want to maximize (offset - mu) / cost
    # This is a cost-aware variant of Thompson Sampling.
    scores = (offset - mu) / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
