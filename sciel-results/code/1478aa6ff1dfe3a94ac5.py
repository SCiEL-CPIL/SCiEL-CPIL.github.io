"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of cosine8 in 8D.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviations
    - f_best: float - lowest observed value
    - cost: array of shape (2000,) - cost of evaluating each point (1 + mean(x_scaled))
    
    Returns:
    - score: array of shape (2000,) - acquisition scores (higher = more promising)
    """
    t_start = time.time()
    
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower bound on objective: -0.8 (minimum of cosine8)
    f_min_possible = -0.8
    
    # Estimate how close we are to the optimum
    improvement_room = f_best - f_min_possible
    
    # Estimate progress from f_best
    f_max_estimated = 1.0
    f_range = f_max_estimated - f_min_possible
    progress = np.clip(1.0 - improvement_room / f_range, 0, 1)
    
    # Multiplicative gate: score = EI * exp(-k * cost)
    # Calibrate k to the expected EI scale
    # EI is in objective units; typical EI values are on the order of sigma or improvement_room
    # cost is in [1, 2]
    # We want exp(-k * cost) to be a meaningful gate:
    #   - When cost is low (~1), the gate should be close to 1
    #   - When cost is high (~2), the gate should suppress significantly unless EI is large
    
    # The scale of EI: let's use the typical sigma scale
    # sigma is in objective units; for cosine8, typical values might be 0.1-0.5
    # We want k * cost to be O(1) for cost in [1, 2]
    
    # Adaptive k: early on, we want more exploration (lower k), late on, more exploitation (higher k)
    # But the multiplicative gate naturally handles this: high EI (exploitation) passes through,
    # while low EI (exploration) is suppressed at high cost
    
    # Calibrate k based on the scale of EI
    # If EI is typically ~0.1, then k ~ 2-5 would give exp(-2) to exp(-5) for cost 1-2
    # Let's use a fixed k that balances exploration and exploitation
    
    # From the knowledge: reducing cost penalty exponent from 1 to 0.5 improved score
    # This suggests we want a softer penalty. exp(-k * cost) with small k is soft.
    
    # Let's use k = 1.0 as a starting point, which gives:
    #   cost=1: exp(-1) ≈ 0.37
    #   cost=2: exp(-2) ≈ 0.14
    # This is a strong penalty. Maybe k = 0.5 is better:
    #   cost=1: exp(-0.5) ≈ 0.61
    #   cost=2: exp(-1) ≈ 0.37
    
    # Let's make k adaptive: lower k when improvement_room is large (early), higher when small (late)
    # Actually, the multiplicative gate already handles this implicitly.
    
    # Use k = 0.5 for a softer penalty, consistent with the finding that 0.5 exponent worked better
    k = 0.5
    
    # Multiplicative gate
    gate = np.exp(-k * cost)
    
    score = ei * gate
    
    # Add a small exploration bonus for high-uncertainty points
    # This helps when EI is zero (mu >= f_best) but sigma is high
    # The bonus should also be gated by cost
    explore_bonus = 0.1 * sigma_safe * gate
    
    score = score + explore_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    t_elapsed = time.time() - t_start
    if t_elapsed > 5.0:
        return score
    
    return score
# EVOLVE-BLOCK-END
