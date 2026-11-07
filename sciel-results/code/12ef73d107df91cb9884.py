import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
TIME_LIMIT = 330.0
SAFETY_MARGIN = 5.0
_eval_count = 0

def acquisition(mu, sigma, f_best):
    global _eval_count
    _eval_count += 1
    
    # Deadline guard: return safe scores if approaching time limit
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT - SAFETY_MARGIN:
        # If we are at the deadline, return a neutral score (e.g., just -mu)
        # to force a selection quickly.
        return -mu
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Probability of Improvement (PI) for minimization
    # PI = Phi(z) where z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    
    # Adaptive exploration: blend EI and PI based on uncertainty
    # When sigma is large relative to the scale of f_best, favor PI for exploration
    # When sigma is small, favor EI for exploitation
    
    # Estimate scale from the data
    mu_scale = np.std(mu) + 1e-10
    sigma_mean = np.mean(sigma_safe)
    
    # Exploration weight: higher when uncertainty is high relative to function scale
    # Use a smooth transition based on the ratio of sigma to mu spread
    exploration_weight = np.clip(sigma_safe / (mu_scale + sigma_mean + 1e-10), 0.0, 1.0)
    
    # Normalize EI to [0, 1] for blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + 1e-10
    ei_normalized = (ei - ei_min) / ei_range
    
    # Blend: score = (1 - w) * EI_normalized + w * PI
    # This gives more weight to PI (exploration) when uncertainty is high
    base_scores = (1.0 - exploration_weight) * ei_normalized + exploration_weight * pi
    
    # Adaptive Beta for LCB component
    # Start with high exploration (beta=4.0) to escape local minima early.
    # Decay to lower exploitation (beta=1.0) as we converge.
    # The decay is exponential with a time constant of 5 evaluations.
    beta = 1.0 + 3.0 * np.exp(-_eval_count / 5.0)
    
    # LCB component for minimization
    lcb_scores = -mu + beta * sigma_safe
    
    # Normalize LCB scores to [0, 1] for blending
    lcb_min = np.min(lcb_scores)
    lcb_max = np.max(lcb_scores)
    lcb_range = lcb_max - lcb_min + 1e-10
    lcb_normalized = (lcb_scores - lcb_min) / lcb_range
    
    # Blend the two strategies: early on favor LCB (exploration), later favor EI/PI blend (exploitation)
    # Use the same adaptive decay for blending weight
    lcb_weight = np.exp(-_eval_count / 10.0)
    
    scores = lcb_weight * lcb_normalized + (1.0 - lcb_weight) * base_scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
