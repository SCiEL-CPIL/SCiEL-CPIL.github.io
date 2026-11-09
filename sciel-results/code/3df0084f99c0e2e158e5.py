"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time to track elapsed wall-clock time
_START_TIME = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    mu: (2000,) mean of GP
    sigma: (2000,) std of GP
    f_best: float, best observed value
    cost: (2000,) cost of evaluating each candidate
    Returns: (2000,) scores, higher is better.
    """
    # Check time limit
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Return a safe fallback: just pick the cheapest point or random
        # Returning zeros is safe but might not improve. 
        # Let's return -mu / cost to at least exploit
        with np.errstate(divide='ignore', invalid='ignore'):
            fallback = -mu / cost
        fallback[~np.isfinite(fallback)] = 0.0
        return fallback

    # Ensure sigma is not zero to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Use scipy.stats.norm for accuracy, but numpy is faster if we approximate?
    # scipy is available and accurate.
    from scipy.stats import norm
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (minimization)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Clip negative EI to 0 (mathematically it's >= 0, but numerically might be -eps)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring
    # The seed uses ei / cost.
    # Let's try a slightly more aggressive cost penalty or an exploration boost.
    # UCB-like term: -mu + beta * sigma
    # Let's blend EI and UCB.
    # score = (ei + 0.5 * sigma_safe) / cost
    # This adds a small exploration bonus to the EI.
    
    # Alternatively, pure UCB is often better for multimodal.
    # Let's try UCB: score = (-mu + 1.5 * sigma_safe) / cost
    # But EI is usually better for convergence.
    
    # Let's stick to EI but ensure it's robust.
    # One issue with EI is it can be "myopic".
    # Let's try: score = ei / (cost * (1 + 0.1 * (sigma_safe / (f_best + 1e-6))))
    # This penalizes high uncertainty more? No, that would reduce exploration.
    
    # Let's try a simple modification:
    # The seed program is `ei / cost`.
    # Maybe the issue is that `cost` is in [1, 2].
    # Let's try `ei / (cost ** 2)` to heavily favor cheap points?
    # Or `ei / cost + 0.1 * sigma_safe / cost`?
    
    # Let's try a UCB variant which is structurally different.
    # Minimization UCB: we want to minimize f(x).
    # Upper bound of f(x) is roughly mu + beta*sigma.
    # We want to minimize this upper bound? No, that's conservative.
    # We want to find where f(x) is LOW.
    # So we want to maximize the probability that f(x) is low.
    # PI: P(f(x) < f_best) = cdf(z).
    # EI: E[f_best - f(x) | f(x) < f_best].
    
    # Let's try a combination:
    # score = (ei + 0.2 * sigma_safe) / cost
    # This encourages exploration of uncertain regions even if EI is low.
    
    score = (ei + 0.2 * sigma_safe) / cost
    
    # Ensure finite
    score[~np.isfinite(score)] = 0.0
    
    return score
# EVOLVE-BLOCK-END
