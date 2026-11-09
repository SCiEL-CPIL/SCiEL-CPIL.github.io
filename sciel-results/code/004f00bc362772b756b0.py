"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_START_TIME = None
_DEADLINE = 320.0  # 10s safety margin before 330s kill

def _init_timer():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.monotonic()

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function with cost awareness.
    
    Samples a single latent function value y ~ N(mu, sigma^2) for each candidate.
    Scores candidates by -y / cost. This naturally balances exploration (via sigma)
    and exploitation (via mu) without tuning hyperparameters, and explicitly
    accounts for evaluation cost.
    
    Parameters:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP std predictions
        f_best: float, best observed value (not directly used in TS score, 
                but influences mu/sigma via GP conditioning)
        cost: (2000,) cost of evaluating each candidate
        
    Returns:
        (2000,) acquisition scores (higher = evaluate next)
    """
    _init_timer()
    
    # Convert inputs to numpy arrays for safety and speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid sampling errors
    # 1e-8 is small enough to not distort the distribution significantly
    # but large enough to prevent division by zero or invalid std
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure cost is strictly positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Check deadline: if we are close to the limit, we might want to simplify.
    # However, Thompson Sampling is computationally very cheap (O(N) sampling).
    # The main cost is in the GP update which happens OUTSIDE this function.
    # So we can proceed with the full TS calculation safely.
    # If time is critical, we could fall back to a deterministic proxy, 
    # but TS is fast. We just need to ensure we don't hang.
    
    # Sample y ~ N(mu, sigma_safe^2)
    # np.random.normal(loc, scale)
    y = np.random.normal(mu, sigma_safe)
    
    # Score: -y / cost
    # We want to minimize f, so lower y is better.
    # We want to maximize score, so we use -y.
    # Higher cost is penalized by dividing by cost.
    score = -y / cost_safe
    
    # Ensure finite values
    # If sigma is very large or mu is extreme, y might be inf/nan
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
