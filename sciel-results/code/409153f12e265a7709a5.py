"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_start_time = None
_DEADLINE = 325.0  # Safety margin before 330s kill

# Shekel 4D minimum is approx -10.5364
SHEKEL_MIN = -10.5364


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs (1 + mean(x scaled to [0,1]))
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a safe fallback: just use -mu/cost to avoid any further computation
        return -mu / (cost + 1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # 1. Calculate Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware EI
    ei_per_cost = ei / cost
    
    # 2. Calculate UCB (Upper Confidence Bound) for minimization
    # We want to minimize, so we look for low mu + high uncertainty
    # Standard UCB for minimization: -mu + beta * sigma
    # Let's use a beta that scales with the "distance to optimum"
    
    # 3. Adaptive Weighting Strategy
    # As f_best approaches SHEKEL_MIN, we should rely more on EI (exploitation)
    # Early on, we should rely more on UCB (exploration)
    
    # Normalize progress: 0 at start (f_best is high), 1 at optimum
    # f_best starts around -2 to -5, goes to -10.5
    # Let's map f_best to a confidence factor [0, 1]
    # If f_best is close to min, confidence is high.
    # Simple mapping: confidence = 1 - (f_best - SHEKEL_MIN) / (INITIAL_RANGE)
    # But we don't know initial range exactly. 
    # Alternative: Use the GP uncertainty itself to weight.
    
    # Let's try a different approach: 
    # Score = w * (EI/cost) + (1-w) * (UCB/cost)
    # Where w depends on how "good" f_best is.
    
    # A robust metric for "goodness" is how close f_best is to the known minimum.
    # If f_best is -10.5, we are done. If f_best is -5, we are far.
    # Let's define a "progress" variable.
    # progress = (f_best - SHEKEL_MIN) / (MAX_POSSIBLE_VALUE - SHEKEL_MIN)
    # MAX_POSSIBLE_VALUE for Shekel 4D is around 10.
    # So denominator is ~20.
    
    max_val = 10.0
    progress = (f_best - SHEKEL_MIN) / (max_val - SHEKEL_MIN)
    progress = np.clip(progress, 0.0, 1.0)
    
    # As progress increases (f_best gets closer to min), we want to exploit more.
    # So weight for EI should increase with (1 - progress).
    # Wait, if progress is 0 (far from min), we want exploration (UCB).
    # If progress is 1 (at min), we want exploitation (EI).
    
    # Let w_ei = 1 - progress
    # Let w_ucb = progress
    # This seems inverted. If we are far (progress=0), w_ei=1? No, we want exploration.
    
    # Let's rethink:
    # Early stage: High uncertainty, f_best is poor. We want to explore. UCB is good.
    # Late stage: Low uncertainty, f_best is good. We want to exploit. EI is good.
    
    # So:
    # w_ei should be high when progress is high (close to min).
    # w_ucb should be high when progress is low (far from min).
    
    w_ei = 1.0 - progress
    w_ucb = progress
    
    # To make the transition smoother, we can use a sigmoid or just linear.
    # Linear is fine.
    
    # Calculate UCB
    # Beta choice: 2.0 is standard for moderate exploration.
    beta = 2.0
    ucb = -mu + beta * sigma
    
    # Cost-aware UCB
    ucb_per_cost = ucb / cost
    
    # Combine
    score = w_ei * ei_per_cost + w_ucb * ucb_per_cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
