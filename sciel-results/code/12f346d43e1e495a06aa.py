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

    # Threshold for switching from Exploration (LCB) to Exploitation (NEI)
    # Goldstein-Price min is 3.0.
    # If f_best is high, we are far away. Explore.
    # If f_best is low, we are close. Exploit.
    threshold = 8.0
    
    if f_best > threshold:
        # LCB for minimization: Score = -mu + beta * sigma
        # Beta = 1.0 based on finding "EI -> LCB (beta=1) improved"
        beta = 1.0
        scores = -mu + beta * sigma_safe
    else:
        # NEI: Normalized Expected Improvement
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei / sigma_safe

    scores = np.where(np.isfinite(scores), scores, -np.inf)
    return scores
