"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses a Thompson Sampling approach with cost-aware normalization to balance
exploration and exploitation while respecting budget constraints.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline management
_START_TIME = time.monotonic()
_DEADLINE_SECONDS = 285.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    Draws a single posterior sample and selects candidates based on
    sampled function value normalized by cost.
    
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
    
    n = len(mu)
    
    # Thompson Sampling: draw a single posterior sample
    # f_s = mu + sigma * z where z ~ N(0, 1)
    rng = np.random.default_rng()
    z = rng.standard_normal(n)
    f_sampled = mu + sigma * z
    
    # Cost-aware normalization: divide by (cost - 0.5) to amplify cost signal
    cost_adjusted = cost - 0.5
    cost_adjusted = np.maximum(cost_adjusted, 0.1)  # Avoid division by zero
    
    # Primary score: sampled function value normalized by cost
    # Lower f_sampled is better, so we negate for maximization
    ts_score = -f_sampled / cost_adjusted
    
    # Add a small Expected Improvement component for stability
    # This helps when Thompson Sampling might be too stochastic
    z_ei = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    ei_cost = ei / cost_adjusted
    
    # Blend Thompson Sampling with EI for robustness
    # Weight: 0.7 for TS, 0.3 for EI
    alpha = 0.7
    score = alpha * ts_score + (1 - alpha) * ei_cost
    
    # Add exploration bonus based on uncertainty
    # Reward moderate uncertainty levels
    sigma_max = np.max(sigma)
    sigma_min = np.min(sigma)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Gaussian-like exploration bonus peaking at moderate uncertainty
    sigma_center = 0.4
    sigma_width = 0.3
    exploration_bonus = 0.1 * np.exp(-0.5 * ((sigma_norm - sigma_center) / sigma_width) ** 2)
    
    score += exploration_bonus
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
