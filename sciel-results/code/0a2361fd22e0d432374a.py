import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Time guard: If we are close to the limit, return a simple score to avoid timeout
    elapsed = time.time() - _start_time
    if elapsed > 325:
        return -mu

    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Hybrid approach: Blend UCB and EI
    # UCB: -mu + beta * sigma
    # EI: (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    
    # For minimization, standard EI is (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Note: This EI is positive when mu < f_best (improvement expected)
    
    # Let's try a weighted combination or a specific parameterization.
    # Inspiration 3 used EI with xi=-0.5 (exploratory).
    # Current program uses UCB with beta=2.5.
    
    # Let's try a dynamic beta or a mix.
    # A common robust strategy is to use UCB but with a beta that adapts or is slightly different.
    # Or, since EI scored lower in inspiration 3 (1.0895 vs 1.1783), maybe pure EI isn't as good here.
    # However, a blend might capture the benefits of both.
    
    # Let's try a "Soft-UCB" or a mix of UCB and EI.
    # Score = w * UCB + (1-w) * EI_scaled
    
    # First, compute EI
    # To avoid division by zero, use a small epsilon
    eps = 1e-9
    z = (f_best - mu) / (sigma + eps)
    
    # EI for minimization:
    # E[max(f_best - f(x), 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If sigma is very small, z becomes large.
    # If mu < f_best, z > 0, Phi(z) ~ 1, EI ~ f_best - mu.
    # If mu > f_best, z < 0, Phi(z) ~ 0, EI ~ 0 (plus small term).
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB
    beta = 2.5
    ucb = -mu + beta * sigma
    
    # Normalize EI to be comparable to UCB?
    # UCB can be negative (if mu is large positive). EI is non-negative.
    # This direct sum might be biased.
    
    # Alternative: Use UCB with a slightly different beta, or add a small EI term for exploration balance.
    # Let's try: score = ucb + 0.1 * ei
    # Or, since EI is already an acquisition function, maybe just try a different beta for UCB?
    # The current beta is 2.5. Maybe 2.0 or 3.0?
    
    # Let's try a mix: 0.8 * UCB + 0.2 * EI
    # But scaling matters.
    
    # Let's look at the scales.
    # mu is in objective units (e.g., around -3).
    # sigma is in objective units (e.g., 0.1 - 1.0).
    # UCB ~ -(-3) + 2.5*0.5 = 3 + 1.25 = 4.25
    # EI ~ ( -3 - (-3) ) * 1 + 0.5 * 0.4 = 0.2
    # The scales are quite different. UCB dominates.
    
    # Perhaps a better strategy is to use UCB but with a beta that accounts for the number of evaluations?
    # Or stick with UCB but tweak beta slightly?
    
    # Given Inspiration 3 (EI) was worse, and Current (UCB) is better, let's stick closer to UCB.
    # Maybe add a small amount of EI to break ties or encourage exploration in flat regions?
    
    # Let's try: score = ucb + 0.05 * ei
    # This adds a small exploratory component from EI without letting it dominate.
    
    scores = ucb + 0.05 * ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
