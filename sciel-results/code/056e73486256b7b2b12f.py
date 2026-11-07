# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid EI/UCB acquisition with z-score normalization and cost awareness.
    Balances exploitation and exploration for high-dimensional multimodal landscapes.
    """
    # Deadline guard
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 325.0:  # Safety margin
        # Fallback: simple heuristic
        return -mu / (np.maximum(cost, 1e-9))

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50.0, 50.0)
    
    # Compute EI
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB term: -mu + beta * sigma
    # Use adaptive beta based on sigma magnitude
    # Larger sigma -> more exploration
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Normalize both terms to comparable scales
    # Use the range of each term for normalization
    ei_range = np.max(ei) - np.min(ei) + 1e-12
    ucb_range = np.max(ucb) - np.min(ucb) + 1e-12
    
    ei_norm = (ei - np.min(ei)) / ei_range
    ucb_norm = (ucb - np.min(ucb)) / ucb_range
    
    # Blend EI and UCB: weight exploration more when uncertainty is high
    # Use the mean sigma to determine the blend ratio
    mean_sigma = np.mean(sigma_safe)
    # If mean sigma is large relative to f_best scale, explore more
    # f_best is typically negative for this problem (min around -0.8)
    sigma_scale = mean_sigma / (abs(f_best) + 1e-9)
    # Clip exploration weight to [0.3, 0.7]
    explore_weight = np.clip(0.3 + 0.4 * min(sigma_scale, 1.0), 0.3, 0.7)
    
    # Combined score before cost adjustment
    combined = (1 - explore_weight) * ei_norm + explore_weight * ucb_norm
    
    # Cost-aware: divide by cost, but use a softer penalty
    # cost = 1 + mean(x_scaled), so cost in [1, 2]
    cost_safe = np.maximum(cost, 1e-9)
    score = combined / np.sqrt(cost_safe)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
