"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function for Bayesian optimization of sphere 1D.
    
    Uses a blend of:
    - Lower Confidence Bound (LCB) for exploitation: mu - beta * sigma
    - Expected Improvement (EI) for balanced exploration/exploitation
    
    The blend weight adapts based on how close we are to the minimum (f_best).
    When f_best is large (early), we explore more. When f_best is small (late),
    we exploit more.
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Lower Confidence Bound (we want to minimize, so we prefer low mu + high uncertainty)
    # LCB score = mu - beta * sigma (lower is better for the objective)
    # But we want higher acquisition score to mean "evaluate next"
    # So we negate: -mu + beta * sigma
    beta_lcb = 2.0
    lcb_score = -mu + beta_lcb * sigma
    
    # Adaptive blending: 
    # When f_best is large (far from optimum), explore more (weight EI more)
    # When f_best is small (close to optimum), exploit more (weight LCB more)
    # Use a sigmoid-like transition based on f_best
    # f_best ranges from ~25 (x=5) down to 0
    # Transition around f_best = 1.0
    transition = 1.0
    width = 2.0
    # w is weight on LCB (exploitation), 1-w is weight on EI (exploration)
    # When f_best >> transition, w -> 0 (more exploration)
    # When f_best << transition, w -> 1 (more exploitation)
    w = 1.0 / (1.0 + np.exp((f_best - transition) / width))
    
    # Normalize EI to be on similar scale as LCB
    # EI is always >= 0, LCB can be negative
    # Scale EI by a factor to make it comparable
    ei_scale = 1.0
    ei_score = ei * ei_scale
    
    # Combine: higher score = evaluate next
    # We want: high exploration when f_best is large, high exploitation when small
    score = w * lcb_score + (1.0 - w) * ei_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
