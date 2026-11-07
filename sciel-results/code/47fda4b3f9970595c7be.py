"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a Stage-Adaptive Hybrid of Expected Improvement (EI) and Lower Confidence Bound (LCB).
    - Early stage (high f_best): Favors LCB for aggressive global exploration.
    - Late stage (low f_best): Favors EI for robust local exploitation.
    - Both terms are normalized by evaluation cost to prioritize cheaper evaluations.
    """
    start_time = time.time()
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Cost normalization: cost is in [1, 2]
    cost_safe = np.maximum(cost, 1e-10)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute LCB Improvement
    # LCB = mu - beta * sigma
    # Improvement = f_best - LCB = f_best - mu + beta * sigma
    # We use a moderate beta for the LCB term.
    beta = 2.5
    lcb_improvement = f_best - mu + beta * sigma_safe
    lcb_improvement = np.maximum(lcb_improvement, 0.0)
    
    # 3. Cost-Weighted Scores
    # We normalize both by cost to strictly respect the budget constraint.
    ei_score = ei / cost_safe
    lcb_score = lcb_improvement / cost_safe
    
    # 4. Stage-Adaptive Weighting
    # Estimate the stage of optimization based on f_best.
    # For 3-hump camel, min is 0. Initial values are likely > 1.
    # We transition from Exploration (LCB) to Exploitation (EI) as f_best decreases.
    # Threshold T=1.0, Slope K=5.0
    # w is the weight for LCB (Exploration). (1-w) is weight for EI (Exploitation).
    T = 1.0
    K = 5.0
    # Sigmoid: if f_best >> T, w -> 1 (High LCB). If f_best << T, w -> 0 (High EI).
    # Avoid overflow in exp
    arg = K * (f_best - T)
    arg = np.clip(arg, -50, 50)
    w_lcb = 1.0 / (1.0 + np.exp(-arg))
    
    # Combine scores
    # Higher w_lcb means more reliance on LCB (aggressive).
    scores = w_lcb * lcb_score + (1.0 - w_lcb) * ei_score
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 280.0:
        # Fallback to simple EI/cost
        simple_scores = ei / cost_safe
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
# EVOLVE-BLOCK-END
