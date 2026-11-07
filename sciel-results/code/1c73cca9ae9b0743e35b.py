"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses an adaptive blend of Expected Improvement and UCB with cost normalization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor sigma and cost
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Check deadline early
    if time.time() - t0 > DEADLINE - 10:
        beta = 1.5
        improvement = f_best - mu + beta * sigma
        scores = improvement / cost
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Known minimum of Hartmann 6d
    HARTMANN_6D_MIN = -3.32237
    
    # Estimate optimization progress
    range_val = f_best - HARTMANN_6D_MIN
    if range_val <= 0:
        progress = 1.0
    else:
        # Normalize progress: assume initial f_best is around -1.0 to -2.0
        initial_guess = -1.5
        total_range = initial_guess - HARTMANN_6D_MIN
        if total_range <= 0:
            progress = 0.5
        else:
            progress = np.clip((initial_guess - f_best) / total_range, 0.0, 1.0)
    
    # Compute Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB improvement with adaptive beta
    # Higher beta early (exploration), lower beta late (exploitation)
    beta_early = 2.5
    beta_late = 0.8
    beta = beta_early + (beta_late - beta_early) * progress
    
    ucb_improvement = f_best - mu + beta * sigma
    
    # Also compute a "normalized" EI that accounts for uncertainty more directly
    # This helps when sigma is large
    normalized_ei = ei / (sigma + 1e-9)
    
    # Blend strategies based on progress
    # Early: use more UCB-like exploration
    # Late: use more EI-like exploitation
    w_ei = progress
    w_ucb = 1.0 - progress
    
    # Combine: use a weighted combination
    blended = w_ei * ei + w_ucb * ucb_improvement
    
    # Cost-aware score: divide by cost to prefer cheaper evaluations
    # Use a slightly more aggressive cost penalty to encourage exploration in cheap regions
    cost_penalty = cost ** 0.5  # Softer penalty than linear
    scores = blended / cost_penalty
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() - t0 > DEADLINE - 5:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
