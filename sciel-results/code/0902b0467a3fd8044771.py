"""Rewritten program: cost-aware Expected Improvement with adaptive exploration for Griewank 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing Griewank 2D using Expected Improvement
    with an adaptive exploration bonus based on uncertainty and cost.
    
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
    DEADLINE = 290.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    cost = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)
    sigma = np.maximum(sigma, 1e-9)
    
    f_best = float(f_best) if np.isfinite(f_best) else 0.0
    
    # Compute z-scores for EI
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Adaptive exploration bonus: reward high-uncertainty, low-cost points
    # Normalize sigma to [0, 1] range for this batch
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min if sigma_max > sigma_min else 1.0
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Normalize cost to [0, 1] range (lower cost = higher value)
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min if cost_max > cost_min else 1.0
    cost_norm = (cost_max - cost) / cost_range  # Invert: lower cost -> higher value
    
    # Exploration bonus: weighted combination of normalized uncertainty and inverse cost
    # The weight adapts: when f_best is close to 0 (minimum), we can afford more exploration
    # When f_best is far from 0, we should focus more on exploitation
    improvement_potential = max(f_best, 0.0)
    exploration_weight = 0.3 + 0.2 * np.tanh(improvement_potential / 0.01)
    
    exploration_bonus = exploration_weight * (0.6 * sigma_norm + 0.4 * cost_norm)
    
    # Combine EI and exploration bonus, both cost-aware
    # EI is already in objective units; scale it to be comparable to exploration bonus
    # Use the mean EI as a reference scale
    ei_mean = np.mean(ei) if np.mean(ei) > 1e-12 else 1e-12
    ei_scaled = ei / ei_mean
    
    # Final score: scaled EI + exploration bonus, divided by cost
    scores = (ei_scaled + exploration_bonus) / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
