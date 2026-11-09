"""
Acquisition function for Bayesian optimization of Ackley 2D.
Implements a Deterministic Entropy-Weighted UCB strategy.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for the deadline guard
_START_TIME = time.time()
# The evaluator kills after 330s. We aim to finish well before.
# However, since this function is called many times, we just check if we are
# extremely close to the limit. But typically, the GP fit is the bottleneck.
# We assume the evaluator handles the total time. 
# If we must guard here, we just return a safe default if time is critical.
DEADLINE_MARGIN = 5.0 # seconds

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores for candidates.
    
    Args:
        mu (np.ndarray): Mean of the GP posterior, shape (N,).
        sigma (np.ndarray): Standard deviation of the GP posterior, shape (N,).
        f_best (float): The best (lowest) function value observed so far.
        
    Returns:
        np.ndarray: Scores for each candidate, shape (N,). Higher is better.
    """
    # Deadline guard: If we are too close to the limit, return a safe, 
    # deterministic result (e.g., exploit best mean) to avoid timeout.
    elapsed = time.time() - _START_TIME
    if elapsed > 325.0: # 5 seconds margin before 330s
        # Fallback: Return negative mean (pure exploitation) to be safe
        mu_safe = np.asarray(mu, dtype=np.float64)
        mu_safe = np.where(np.isfinite(mu_safe), mu_safe, 0.0)
        return -mu_safe

    # Convert inputs to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize inputs
    # Replace non-finite mu with 0.0 (neutral)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Replace non-finite sigma with a small positive value
    # Sigma must be positive for the log term and UCB logic
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    # Ensure sigma is strictly positive to avoid log(0) issues or negative sigma
    sigma = np.maximum(sigma, 1e-6)
    
    # Sanitize f_best
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 1.0
        
    # Dynamic exploration weight gamma
    # Early in search: f_best is large (e.g., > 1.0) -> gamma = 1.0 (High Exploration)
    # Late in search: f_best is small (e.g., < 0.1) -> gamma = 0.1 (High Exploitation)
    # The Ackley function has a minimum of 0.
    gamma = np.clip(f_best, 0.1, 1.0)
    
    # Compute Entropy-Weighted UCB Score
    # Term 1: -mu (Exploitation: prefer low mean)
    # Term 2: gamma * sigma * log(1 + sigma) (Exploration: prefer high uncertainty,
    #         but with a logarithmic dampening to prevent extreme jumps to 
    #         high-variance regions if they are not promising)
    
    # log(1 + sigma) is numerically stable for sigma >= 0
    log_term = np.log1p(sigma)
    
    scores = -mu + gamma * sigma * log_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
