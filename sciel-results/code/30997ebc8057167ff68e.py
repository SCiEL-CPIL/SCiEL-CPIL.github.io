"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using Bayesian Regret Minimization (BRM).
    
    Structural change from pointwise EI: uses global softmax ranking over all candidates
    to estimate posterior probability of being the new minimum, then combines with
    cost-awareness. This captures the joint structure of the candidate set rather than
    treating each point independently.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best observed value so far (lowest, since we minimize)
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x scaled to [0,1])
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    # Deadline guard
    t0 = time.time()
    deadline = 295.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # BRM core: compute posterior probability of each candidate being the new minimum
    # P(x is min | data) ∝ exp(-(f_best - mu_i) / (sigma_i + eps))
    # This is a softmax over the "improvement potential" scaled by uncertainty
    
    # Compute improvement potential: how much better could this point be?
    improvement = f_best - mu  # positive if mu < f_best
    
    # Use a smooth temperature based on median sigma for stable softmax
    # This decouples the temperature from the spread of improvements
    temp = np.median(sigma_safe) + 1e-9
    z_improve = improvement / temp
    
    # Softmax over z_improve to get probability weights
    # Subtract max for numerical stability
    z_max = np.max(z_improve)
    exp_z = np.exp(z_improve - z_max)
    prob_min = exp_z / (np.sum(exp_z) + 1e-12)
    
    # Cost-aware BRM: use smooth cost-awareness instead of hard division
    # This avoids extreme scores when cost varies significantly
    # cost_factor in (0, 1], higher cost -> lower factor
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min + 1e-12
    # Smooth mapping: cost_factor = 1 - 0.5 * normalized_cost
    # This gives cost_factor in [0.5, 1.0] for cost in [cost_min, cost_max]
    cost_norm = (cost_safe - cost_min) / cost_range
    cost_factor = 1.0 - 0.5 * cost_norm
    brm_score = prob_min * cost_factor
    
    # Exploration component: reward high-uncertainty regions with a sigmoid-like shape
    # This creates a smooth transition from exploitation to exploration
    sigma_median = np.median(sigma_safe)
    # Explore when sigma is high relative to the median
    explore_raw = sigma_safe / (sigma_median + 1e-9)
    # Sigmoid-like smooth cap: 1 / (1 + exp(-k * (x - 1)))
    k_explore = 2.0
    explore_smooth = 1.0 / (1.0 + np.exp(-k_explore * (explore_raw - 1.0)))
    
    # Scale exploration by BRM score magnitude for balance
    brm_max = np.max(brm_score) if np.max(brm_score) > 1e-15 else 1.0
    exploration = 0.3 * explore_smooth * brm_max
    
    # Combine BRM and exploration
    scores = brm_score + exploration
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    if time.time() - t0 > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
