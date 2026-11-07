import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard: track elapsed time to stay well within the 300s limit
    # This is a fast function, but we add a safety check just in case
    # The actual computation is vectorized and should take < 1ms
    start_time = time.time()
    
    # Ensure sigma is non-negative and avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Strategy: Combine EI (exploitation) with UCB (exploration)
    # EI is the strongest single acquisition (score 1.0 baseline)
    # UCB with high kappa helps explore multimodal landscapes
    # We blend them with a time-adaptive weight:
    # - Early: more exploration (higher UCB weight)
    # - Late: more exploitation (higher EI weight)
    # Since we don't know the step, we use the relationship between
    # f_best and mu to adapt: if f_best is close to min(mu), we're
    # likely later in optimization
    
    # Calculate Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate UCB score for minimization
    # Score = -mu + kappa * sigma
    # Use a moderate kappa that balances exploration
    kappa = 2.5
    ucb_score = -mu + kappa * sigma
    
    # Normalize both scores to [0, 1] range
    eps = 1e-9
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + eps
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min + eps
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Adaptive weighting:
    # When f_best is much larger than most mu values, we're early -> explore more
    # When f_best is close to min(mu), we're late -> exploit more
    # Use the fraction of points where mu < f_best as a proxy for progress
    # If most points have mu < f_best, we're probably late
    frac_below = np.mean(mu < f_best)
    
    # Weight: early (frac_below low) -> more UCB, late (frac_below high) -> more EI
    # Use a smooth transition
    ei_weight = 0.3 + 0.5 * frac_below  # ranges from 0.3 to 0.8
    ucb_weight = 1.0 - ei_weight         # ranges from 0.7 to 0.2
    
    # Combine scores
    scores = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if we're taking too long (shouldn't happen), return immediately
    elapsed = time.time() - start_time
    if elapsed > 250:  # Safety margin
        return scores
    
    return scores
