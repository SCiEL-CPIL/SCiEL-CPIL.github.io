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
    
    # Probability of Improvement (PI)
    # This is the probability that the predicted value at a candidate point
    # is better (lower) than the current best f_best.
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    pi = norm.cdf(z)
    
    # We return the PI score. Higher score means higher probability of improvement.
    # This naturally balances exploitation (low mu) and exploration (high sigma)
    # because high sigma increases the chance that the tail of the distribution
    # extends below f_best, but the effect is dampened compared to UCB which
    # linearly rewards high sigma.
    
    scores = pi
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
