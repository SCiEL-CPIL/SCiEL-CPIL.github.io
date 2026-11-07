"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: balance exploitation (EI) and exploration (UCB)
    # with adaptive exploration parameter and cost normalization.
    # 
    # Strategy:
    # 1. Compute EI (expected improvement) for exploitation
    # 2. Compute UCB (upper confidence bound) for exploration  
    # 3. Blend them with an adaptive beta that decreases as we get more confident
    # 4. Normalize by cost to prefer cheap points
    #
    # The key insight from the island that improved to 1.122:
    # Adaptive UCB provides better exploration-exploitation trade-off than pure EI
    # in small budgets, avoiding local optima traps.
    
    # Ensure valid inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (for minimization: we want mu < f_best)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (for minimization: lower is better, so we use lower bound)
    # LCB = mu - beta * sigma
    # Adaptive beta: start high for exploration, decrease as we have more data
    # Use a schedule based on the cost budget - more cost means more data collected
    # cost ranges from ~1 to ~30, so normalize
    cost_norm = np.clip(cost, 1.0, 30.0)
    
    # Beta schedule: high exploration early, low exploration later
    # beta = sqrt(2 * log(t)) / t style, but tuned for our budget
    # At cost=5 (start), beta ~ 2.0; at cost=30 (end), beta ~ 0.5
    beta = 2.0 * np.exp(-0.1 * (cost_norm - 5.0)) + 0.3
    
    lcb = mu - beta * sigma_safe
    
    # Blend EI and LCB: weight toward exploitation as we get more confident
    # Early on: more exploration (LCB weight higher)
    # Late on: more exploitation (EI weight higher)
    # Use a smooth transition
    # progress = (cost - 5) / 25, clamped to [0, 1]
    progress = np.clip((cost_norm - 5.0) / 25.0, 0.0, 1.0)
    
    # EI is in objective units, LCB is also in objective units
    # We want a score where higher = better (more likely to be next evaluation)
    # Both EI and -LCB are "improvement" measures
    ei_score = ei
    lcb_score = -lcb  # higher is better (more negative mu means better)
    
    # Blend: early on favor LCB (exploration), later favor EI (exploitation)
    w_explore = 1.0 - progress
    w_exploit = progress
    
    # Normalize both to comparable scales
    # EI can be zero or negative, LCB score can be anything
    # Use a soft combination
    score = w_exploit * ei_score + w_explore * lcb_score
    
    # Cost-aware normalization: divide by cost to prefer cheap points
    # But don't divide by zero
    cost_safe = np.maximum(cost, 1e-9)
    score = score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
