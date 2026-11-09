"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function for minimizing Griewank 2D.
    
    PI provides a bounded, sigmoidal exploration signal that naturally saturates in high-uncertainty
    regions, avoiding unbounded noise-chasing. Combined with cost normalization, this yields a
    well-behaved acquisition landscape for sparse-data Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best (lowest) observed value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    DEADLINE = 290.0  # Safety margin before 300s limit
    
    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # f_best is the best observed value (lowest since we minimize)
    f_best = float(f_best) if np.isfinite(f_best) else 0.0
    
    # Compute Z-score: z = (f_best - mu) / sigma
    # Positive z means mu < f_best (potential improvement)
    z = (f_best - mu) / sigma
    
    # Clip Z to avoid numerical issues in CDF evaluation
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    # This is bounded in [0, 1], providing a natural exploration/exploitation tradeoff
    pi = norm.cdf(z)
    
    # Cost-aware score: PI / cost
    # Normalizing by cost ensures expensive points are penalized appropriately
    scores = pi / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
