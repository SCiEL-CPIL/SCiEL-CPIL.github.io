"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track progress
# We assume the function is called sequentially for a single run.
# Since we don't know the total number of calls in advance (25 BO steps),
# we can use a simple heuristic or just rely on f_best improvement.
# However, a robust approach for this specific short horizon (25 steps) 
# is to use a fixed schedule based on the number of calls, but we don't have that.
# Instead, we can use a heuristic based on f_best.
# But the simplest robust "different" strategy is to change the exploration 
# constant dynamically based on the spread of sigma or just use a stronger 
# exploration term than the current portfolio's average.

# Let's try a "Late Exploration" strategy:
# Early on, we want to be sure we find a decent minimum.
# Late on, we want to explore to see if we can beat it.
# Since we can't track step count reliably without global state (which might persist 
# across seeds if not cleared, but the prompt says "For each seed... the evaluator starts"),
# we should assume state is reset or use a stateless heuristic.

# Let's use a stateless heuristic: 
# If f_best is still high (far from 0), exploit more.
# If f_best is low (close to 0), explore more.
# Ackley min is 0.
# Typical f_best starts high (~30) and drops.

def acquisition(mu, sigma, f_best):
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PI) - Exploitation
    pi = norm.cdf(z)
    
    # Expected Improvement (EI) - Balanced
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) - Exploration
    # For minimization, we want low mu and high sigma.
    # Score = -mu + beta * sigma
    # Let's try a dynamic beta.
    
    # Heuristic: 
    # If f_best is large (e.g., > 1.0), we are likely far from the optimum.
    # We should exploit to get closer. Use PI or low-UCB.
    # If f_best is small (e.g., < 1.0), we are close.
    # We should explore to find if we can get to 0. Use high-UCB.
    
    # Threshold for "close"
    threshold = 0.5 
    
    if f_best > threshold:
        # Exploit: Maximize probability of improvement
        # This is equivalent to minimizing mu - small * sigma
        # Or just using PI which is monotonic with (f_best - mu)/sigma
        # Let's use a mild UCB to avoid sticking to local minima too early
        beta = 1.0
    else:
        # Explore: Maximize uncertainty while penalizing high mean
        # Stronger exploration
        beta = 3.0
        
    ucb = -mu + beta * sigma_safe
    
    # Normalize scores to ensure they are in a comparable range
    # The return value is just a score. Higher is better.
    # We can return the raw UCB score.
    
    scores = ucb
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
