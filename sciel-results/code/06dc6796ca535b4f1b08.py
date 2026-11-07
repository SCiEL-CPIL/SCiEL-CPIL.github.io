"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses Thompson Sampling with cost-awareness. Instead of computing expected improvement
pointwise, we draw a single posterior sample f_s = mu + sigma * z and select the
candidate that minimizes f_s / (cost - 0.5). This models the objective uncertainty
as a latent function, allowing the search to exploit high-variance regions that
pure EI might underweight, while the cost denominator biases toward cheaper points.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline management
_START_TIME = time.monotonic()
_DEADLINE_SECONDS = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check deadline
    elapsed = time.monotonic() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure cost is within expected range [1, 2]
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: draw a single posterior sample
    # f_s = mu + sigma * z where z ~ N(0, 1)
    # We use a fixed random draw per call to be deterministic given the same inputs
    # Actually, for Thompson Sampling we want randomness. But since the evaluator
    # calls this function repeatedly, we should use a fresh random draw each time.
    # However, we need to be careful: the function is called once per iteration
    # with 2000 candidates, and we score all 2000. The "best" one is selected.
    # So we draw one z vector of shape (2000,) and compute f_s for all candidates.
    
    z = np.random.standard_normal(mu.shape)
    f_s = mu + sigma * z
    
    # Cost-awareness: divide by (cost - 0.5) to bias toward cheaper points
    # cost is in [1, 2], so cost - 0.5 is in [0.5, 1.5]
    # Lower cost -> smaller denominator -> higher score (for minimization)
    # But we want to minimize f_s, so we want to maximize -f_s / (cost - 0.5)
    # or equivalently minimize f_s / (cost - 0.5)
    # Since we return scores where higher = evaluate next, we negate.
    
    denom = cost - 0.5
    # Ensure denominator is positive and not too small
    denom = np.maximum(denom, 0.1)
    
    # Score: we want to pick the candidate with lowest f_s / (cost - 0.5)
    # So score = -f_s / (cost - 0.5), higher score = better
    score = -f_s / denom
    
    # Add a small exploitation bonus based on f_best to guide toward good regions
    # This helps when the GP is uncertain everywhere
    # Blend: primarily Thompson sampling, but with a slight pull toward f_best
    # Actually, let's keep it pure Thompson Sampling with cost adjustment.
    # The GP posterior should already be informed by f_best through the training.
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
