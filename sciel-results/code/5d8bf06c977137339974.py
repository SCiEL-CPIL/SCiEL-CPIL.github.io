"""Cost-aware adaptive UCB acquisition with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE_START = time.time()
_HARD_LIMIT = 330.0 - 60.0


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    if (time.time() - _DEADLINE_START) > _HARD_LIMIT:
        return np.nan_to_num(-mu / np.maximum(cost, 1e-6), nan=0.0, posinf=0.0, neginf=0.0)

    cost_safe = np.maximum(cost, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive beta: start high for exploration, decrease as f_best improves
    # f_best starts high (worst case) and decreases toward 0 (optimum)
    # Scale beta inversely with how close we are to the optimum
    # Use a smooth decay: beta = base_beta / (1 + f_best^2)
    # When f_best is large (early), beta ~ base_beta
    # When f_best is small (late), beta ~ base_beta / (1 + small) ~ base_beta
    # Actually, let's use: beta = base_beta * exp(-f_best / scale)
    # This gives more exploration early when f_best is large
    
    base_beta = 0.8
    scale = 2.0  # tuning: how quickly beta decays
    beta = base_beta * np.exp(-max(f_best, 0.0) / scale)
    
    # UCB: -mu + beta * sigma (we want to minimize, so lower mu is better)
    ucb = -mu + beta * sigma
    
    # Cost-aware: divide by cost to prefer cheaper points
    # But also add a small EI component for exploitation
    z = np.clip((f_best - mu) / sigma_safe, -1e4, 1e4)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    ei = np.maximum(ei, 0.0)
    
    # Combine UCB and EI with adaptive weights
    # Early on: more UCB (exploration), later: more EI (exploitation)
    w_ei = 0.3 + 0.4 * (1.0 - np.exp(-max(f_best, 0.0) / scale))
    w_ucb = 1.0 - w_ei
    
    # Scale EI to be comparable to UCB
    # EI is in objective units, UCB is in objective units (negative mu + beta*sigma)
    # They should be on similar scales
    
    combined = w_ucb * ucb + w_ei * ei
    
    score = combined / cost_safe
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
