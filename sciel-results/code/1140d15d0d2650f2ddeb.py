"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition blending EI and UCB with adaptive weighting
    # Deadline guard: return early if we're close to timeout
    start_time = time.time()
    deadline = 295.0  # safety margin before 300s limit
    
    # Compute z for EI
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0)
    
    # UCB term
    beta = 1.0 + 2.0 * (f_best / (abs(f_best) + 1e-9))  # adaptive exploration
    ucb = mu + beta * sigma_safe
    
    # Cost-aware normalization: use mean cost for stability
    mean_cost = np.mean(cost)
    cost_norm = cost / (mean_cost + 1e-9)
    
    # Blend EI and UCB based on f_best (more exploration when f_best is high)
    # f_best is negative or zero for ackley (min is 0), so use magnitude
    f_best_abs = abs(f_best)
    weight_ei = 1.0 / (1.0 + f_best_abs * 0.5)
    weight_ucb = 1.0 - weight_ei
    
    # Combined score
    ei_score = ei / cost_norm
    ucb_score = (-ucb) / cost_norm  # minimize ucb means lower is better, so negate
    
    score = weight_ei * ei_score + weight_ucb * ucb_score
    
    # Check deadline
    if time.time() - start_time > deadline:
        # Return best available result
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure finite output
    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
# EVOLVE-BLOCK-END
