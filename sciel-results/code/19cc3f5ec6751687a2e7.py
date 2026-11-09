"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with an adaptive exploration bonus.
    
    The EI term drives exploitation of promising regions, while the
    exploration bonus encourages sampling of uncertain areas. The weight
    of exploration adapts to the current best value found: when f_best is
    high (early in the search), more exploration is encouraged; as f_best
    decreases (late in the search), exploitation dominates.
    
    Both terms are normalized by evaluation cost to prefer cheap evaluations.
    """
    # Deadline guard: track elapsed time
    t0 = time.time()
    
    # Ensure arrays are 1D and handle edge cases
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64).ravel()
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in extreme tails
    z_clipped = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Cost-aware EI: EI per unit cost
    score_ei = ei / cost_safe
    
    # Adaptive exploration weight based on f_best
    # When f_best is high (early), w is high -> more exploration
    # When f_best is low (late), w is low -> more exploitation
    # Using a smooth decay: w = 1 / (1 + f_best / scale)
    # scale = 5.0 gives reasonable behavior for Levy 2D (range ~0 to 100+)
    scale = 5.0
    w = 1.0 / (1.0 + max(f_best, 0.0) / scale)
    
    # Normalize sigma to [0, 1] for the exploration term
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-9:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Exploration bonus: encourages visiting uncertain regions
    # Scaled by w (adaptive) and a fixed coefficient to balance with EI
    # The coefficient 0.15 is chosen so that exploration is meaningful
    # but doesn't dominate the EI signal
    exploration_bonus = 0.15 * w * sigma_norm
    
    # Combine: EI per cost + exploration bonus
    # Note: exploration_bonus is not divided by cost because it represents
    # intrinsic value of uncertainty, not expected improvement per cost
    # However, to maintain cost-awareness, we scale it slightly by 1/cost
    # but keep it dominant enough to matter
    # Actually, let's keep exploration independent of cost to ensure
    # we always explore some, but weight it down overall
    score = score_ei + exploration_bonus
    
    # Ensure all values are finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline check: if we're running out of time, return immediately
    elapsed = time.time() - t0
    if elapsed > 250:  # Safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
