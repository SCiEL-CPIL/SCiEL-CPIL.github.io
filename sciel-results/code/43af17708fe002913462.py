import numpy as np
from scipy.stats import norm
import time

# Global state to track the number of acquisition calls
# This allows us to implement a schedule without passing step info.
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # The evaluator starts with 5 random points, then calls acquisition 25 times.
    # So _call_count will range from 1 to 25 in the BO phase.
    
    # Define a schedule for the exploration-exploitation trade-off.
    # Weierstrass is multimodal. Early on, we need to sample broadly to find the basin.
    # Later, we need to converge to the minimum within the best basin.
    
    if _call_count <= 5:
        # Early phase: Strong exploration to cover the domain
        beta = 4.0
        use_ei = False
    elif _call_count <= 15:
        # Mid phase: Balanced
        beta = 2.0
        use_ei = False
    else:
        # Late phase: Exploitation to refine the best candidate
        # We switch to pure EI (or very low beta UCB) to squeeze out the last bits of regret.
        # EI is generally better for minimization when the posterior is tight.
        use_ei = True

    # Calculate EI
    # z = (f_best - mu) / sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: For minimization, Improvement is f_best - f(x).
    # If f(x) < f_best, Improvement > 0.
    # mu is expected f(x).
    # So z = (f_best - mu) / sigma.
    # If mu < f_best, z > 0.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Safe division
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate UCB for minimization
    # We want to maximize the score.
    # UCB_score = -mu + beta * sigma
    # Higher score = lower mu (good) or higher sigma (uncertain).
    ucb = -mu + beta * sigma
    
    if use_ei:
        score = ei
    else:
        score = ucb

    # Ensure finite scores
    score[~np.isfinite(score)] = -np.inf
    
    # If all scores are -inf (shouldn't happen with safe sigma), fallback to random
    if np.all(np.isneginf(score)):
        score = np.random.rand(len(mu))
        
    return score
