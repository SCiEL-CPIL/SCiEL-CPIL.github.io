import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses Thompson Sampling with cost normalization:
    - Sample a scalar offset from Gaussian(f_best, max_sigma * scale)
    - Score each candidate as -(mu - offset) / cost
    - This stochastically balances exploration and exploitation
    """
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clamp sigma to be non-negative and finite
    sigma = np.clip(sigma, 0.0, None)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Thompson Sampling: sample a single scalar offset
    # The offset represents a "virtual best" that we're trying to beat
    # Sample from N(f_best, max_sigma * scale)
    # Scale controls exploration: higher = more exploration
    
    # Use max sigma as a measure of overall uncertainty in the candidate set
    max_sigma = np.max(sigma) if len(sigma) > 0 else 0.0
    
    # Scale factor: start with moderate exploration, reduce as f_best improves
    # f_best / 29 ≈ 1.0 initially, ≈ 0.0 at optimum
    f_max_ackley = 29.0
    f_ratio = np.clip(f_best / f_max_ackley, 0.0, 1.0)
    
    # Exploration scale: higher when f_best is high (early), lower when f_best is low (late)
    # Use a smooth transition
    exploration_scale = 0.5 + 1.5 * f_ratio  # in [0.5, 2.0]
    
    # Sample the offset
    # Use a fixed seed per call would be wrong; we need randomness
    # But we want reproducibility within a single call for the 2000 candidates
    # The randomness comes from sampling once per acquisition call
    std_offset = max_sigma * exploration_scale
    if std_offset < 1e-10:
        std_offset = 1e-6
    
    # Sample from normal distribution
    offset = np.random.normal(loc=f_best, scale=std_offset)
    
    # Score: -(mu - offset) / cost
    # This is equivalent to: (offset - mu) / cost
    # Higher score = better (we want to maximize this)
    score = (offset - mu) / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, -1e6)
    
    return score
