"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = 315.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid acquisition: blends Expected Improvement (EI) with a UCB exploration bonus.
    
    Structural change: Instead of a pure UCB (which can be too greedy or too exploratory
    depending on beta), we use EI for exploitation (known improvement potential) and
    add a scaled UCB term for exploration. The cost penalty is centered and applied
    to the final score.
    
    Score = alpha * EI_per_cost + (1 - alpha) * (UCB - cost_penalty)
    
    This allows the algorithm to exploit when EI is high (near known good points)
    and explore when EI is low but uncertainty is high (unexplored regions).
    """
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: favor low cost and high uncertainty
        sigma_safe = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
        cost_safe = np.maximum(np.asarray(cost, dtype=np.float64), 1e-9)
        return -cost_safe + 0.1 * sigma_safe

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Safe versions for computation
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Expected Improvement (EI)
    # For minimization, EI = E[max(f_best - f(x), 0)]
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in cdf/pdf
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI by cost to get "improvement per unit cost"
    ei_per_cost = ei / cost_safe
    
    # 2. UCB Term
    # UCB = -mu + beta * sigma
    # We want to maximize this (minimize mu, maximize sigma)
    
    # Adaptive beta based on uncertainty
    max_sigma = np.max(sigma_safe)
    # If max_sigma is high, we are in early stages -> higher beta
    # If max_sigma is low, we are in late stages -> lower beta
    # Scale: beta ranges from 1.0 to 3.0
    beta = 1.0 + 2.0 * min(max_sigma / 1.0, 1.0)
    
    ucb_term = -mu + beta * sigma_safe
    
    # 3. Cost Penalty
    # Center cost penalty to avoid bias
    cost_mean = np.mean(cost_safe)
    # Penalty is subtracted, so higher cost -> lower score
    # We want to penalize high cost relative to the mean
    cost_penalty = 0.5 * (cost_safe - cost_mean)
    
    # 4. Combine
    # Blend EI and UCB
    # Use a fixed blend ratio that has worked well in similar tasks
    # EI is good for exploitation, UCB for exploration
    # Let's try 50% EI and 50% UCB, both adjusted for cost
    
    # Normalize EI and UCB to comparable scales?
    # EI is in objective units (improvement). UCB is in objective units (predicted min).
    # They are already in the same units (objective value).
    # However, EI is always positive (or zero), while UCB can be negative.
    # Let's just sum them with weights.
    
    # Actually, EI_per_cost and UCB are not directly comparable if we divide EI by cost.
    # Let's keep EI raw and UCB raw, and apply cost penalty to both?
    # Or just apply cost penalty to the final score.
    
    # Score = w_ei * EI + w_ucb * UCB - cost_penalty
    # Let's try w_ei = 1.0, w_ucb = 0.5
    # This emphasizes exploitation (EI) but allows exploration (UCB)
    
    scores = 1.0 * ei + 0.5 * ucb_term - cost_penalty
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
