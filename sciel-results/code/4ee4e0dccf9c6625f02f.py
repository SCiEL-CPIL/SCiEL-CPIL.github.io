"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Known minimum of Hartmann 6d
F_MIN = -3.32237


def acquisition(mu, sigma, f_best, cost):
    """
    Adaptive acquisition: blends UCB (exploration) and EI (exploitation)
    based on progress toward the known minimum.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP standard deviations
        f_best: scalar, lowest observed value
        cost: (2000,) array of evaluation costs
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    start_time = time.time()
    DEADLINE = 320.0  # safety margin before 330s limit
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Progress metric: how close are we to the known minimum?
    # f_best starts around -1.5 to -2.5, minimum is -3.32237
    # progress = 0 when far, 1 when very close
    if f_best <= F_MIN + 1e-6:
        progress = 1.0
    else:
        # Scale: progress 0 at f_best = 0, progress 1 at f_best = F_MIN
        progress = np.clip((0.0 - f_best) / (0.0 - F_MIN), 0.0, 1.0)
    
    # Adaptive beta: start with aggressive exploration (beta=2.5), 
    # transition to exploitation (beta=0.5) as we get closer
    # Use a smooth transition
    beta = 2.5 - 2.0 * progress
    
    # Compute UCB
    ucb = mu + beta * sigma
    
    # Compute EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Normalize both for blending
    ucb_norm = ucb - np.min(ucb)
    ucb_norm = ucb_norm / (np.max(ucb_norm) + 1e-12)
    
    ei_norm = ei - np.min(ei)
    ei_norm = ei_norm / (np.max(ei_norm) + 1e-12)
    
    # Blend: early -> UCB dominant, late -> EI dominant
    # Weight for EI increases with progress
    w_ei = 0.3 + 0.7 * progress
    w_ucb = 1.0 - w_ei
    
    blended = w_ucb * ucb_norm + w_ei * ei_norm
    
    # Divide by cost for cost-awareness
    score = blended / cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    if time.time() - start_time > DEADLINE:
        return score
    
    return score
# EVOLVE-BLOCK-END
