"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time and enforce a safety margin
_start_time = None

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    elapsed = _get_elapsed()
    
    # Safety margin: if we're getting close to the 330s limit, return a simple score
    if elapsed > 320:
        z = (f_best - mu) / (sigma + 1e-9)
        return ((f_best - mu) * 0.5 * (1 + np.tanh(z)) + sigma * np.exp(-0.5 * z**2)) / cost
    
    # Compute z-score
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Probability of improvement (PI) using normal CDF approximation
    pi = 0.5 * (1 + np.tanh(0.7978845608028654 * z))
    
    # Expected improvement
    ei = (f_best - mu) * pi + sigma * np.exp(-0.5 * z**2)
    
    # UCB-style exploration: mu - beta * sigma (for minimization, we want low mu)
    # Use a moderate beta that balances exploitation and exploration
    beta = 1.0
    ucb = -mu - beta * sigma  # Negate because we're minimizing
    
    # Cost-aware: use a soft cost penalty
    # cost is in [1, 2], so cost_factor in [1, 2]
    cost_factor = cost
    
    # Structural change: Use a hybrid that combines:
    # 1. UCB for exploration (avoids over-reliance on f_best calibration)
    # 2. PI for probability-based exploitation
    # 3. EI for magnitude-based exploitation
    # Weight them to balance short-budget BO challenges
    
    # Normalize each component to comparable scale
    pi_norm = pi  # already in [0, 1]
    ei_norm = ei / (f_best + 1e-9)  # scale EI by f_best
    ucb_norm = ucb / (np.std(mu) + 1e-9)  # scale UCB by std of mu
    
    # Blend: emphasize PI and UCB, less EI (EI can be miscalibrated early)
    score = (0.5 * pi_norm + 0.3 * ucb_norm + 0.2 * ei_norm) / cost_factor
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
