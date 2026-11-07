"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()


def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Lower Confidence Bound (LCB)
    # LCB = mu - beta * sigma
    # We use a fixed beta for simplicity, or could adapt it. 
    # A common choice is beta ~ 2-3. Let's use 2.5.
    beta = 2.5
    lcb = mu - beta * sigma
    
    # 3. Normalization
    # We want to combine them. Since they have different scales, we normalize.
    # A robust way is to map them to [0, 1] or z-score them.
    # However, z-scoring requires knowing the distribution across candidates, which we have (2000 points).
    # Let's z-score both EI and LCB scores.
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        ei_std = 1.0
    ei_norm = (ei - ei_mean) / ei_std
    
    lcb_mean = np.mean(lcb)
    lcb_std = np.std(lcb)
    if lcb_std < 1e-9:
        lcb_std = 1.0
    lcb_norm = (lcb - lcb_mean) / lcb_std
    
    # 4. Combine
    # Weighted sum. The knowledge suggests a hybrid prevents LCB scale from dominating.
    # Let's use equal weights or slightly favor exploitation (EI) in later stages?
    # Without step count, we can't easily adapt weights dynamically based on progress.
    # However, we can use a fixed weight that balances exploration and exploitation.
    # Let's try 0.5 * EI + 0.5 * LCB.
    # Or perhaps 0.6 * EI + 0.4 * LCB?
    # Let's stick to 0.5/0.5 as a robust starting point for a "normalized hybrid".
    
    score = 0.5 * ei_norm + 0.5 * lcb_norm
    
    # Safety: ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
