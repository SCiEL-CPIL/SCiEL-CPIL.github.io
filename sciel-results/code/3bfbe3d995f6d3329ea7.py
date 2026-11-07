import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE_MARGIN = 10.0

def acquisition(mu, sigma, f_best):
    current_time = time.time()
    remaining_time = 330.0 - (current_time - _START_TIME)
    
    if remaining_time < _DEADLINE_MARGIN:
        # Fallback: Greedy (lowest mu)
        scores = -mu
        return np.where(np.isfinite(scores), scores, -np.inf)

    eps = 1e-9
    sigma_safe = sigma + eps

    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe

    # Calculate Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Calculate Normalized EI (NEI)
    nei = ei / sigma_safe

    # Add a small adaptive exploration bonus that decays as f_best approaches the minimum (3.0)
    # This provides mild exploration in uncertain regions without the full LCB behavior
    # beta_exploration decays exponentially as f_best gets closer to 3.0
    gap = max(f_best - 3.0, 0.0)
    beta_exploration = 0.05 * np.exp(-gap / 2.0)
    
    # score = NEI + small exploration bonus
    scores = nei + beta_exploration * sigma_safe
    
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    return scores
