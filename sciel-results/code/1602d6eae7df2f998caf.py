"""Cost-aware acquisition function for Bayesian optimization (three hump camel 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a stage-adaptive hybrid combining:
    1. Expected Improvement (EI) normalized by cost
    2. Uncertainty-weighted exploration term (LCB-like)
    
    The blend weight shifts from exploration to exploitation as f_best decreases.
    A sqrt-based beta schedule provides stronger early exploration.
    """
    start_time = time.time()
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-10, None)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid overflow
    z_clipped = np.clip(z, -50, 50)
    
    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-10)
    
    # EI per unit cost
    score_ei = ei / cost_safe
    
    # LCB for minimization: we want to maximize -(mu - beta*sigma) = beta*sigma - mu
    # Use a sqrt-based beta schedule: beta starts high when f_best is large
    # beta = 1 + sqrt(max(f_best, 0)) / scale
    # When f_best is 0 (at optimum), beta = 1 (pure exploitation via EI)
    # When f_best is large, beta is larger (more exploration)
    f_best_clipped = np.maximum(f_best, 0.0)
    beta = 1.0 + np.sqrt(f_best_clipped) / 5.0
    lcb_score = beta * sigma_safe - mu
    
    # LCB per unit cost
    score_lcb = lcb_score / cost_safe
    
    # Stage-adaptive weight:
    # w_ei = 1 when f_best <= 0 (late, pure exploitation)
    # w_ei = 0 when f_best >= 50 (early, pure exploration)
    # Use a smooth sigmoid-like transition
    w_ei = 1.0 - f_best / 50.0
    w_ei = np.clip(w_ei, 0.0, 1.0)
    
    # Apply a smoother transition using sigmoid
    # Map f_best from [0, 50] to [0, 1] then apply sigmoid
    x = f_best / 25.0  # center at f_best=25
    w_ei_smooth = 1.0 / (1.0 + np.exp(x))
    # Blend between linear and smooth
    w_ei = 0.5 * w_ei + 0.5 * w_ei_smooth
    w_ei = np.clip(w_ei, 0.0, 1.0)
    
    w_lcb = 1.0 - w_ei
    
    # Combine scores
    scores = w_ei * score_ei + w_lcb * score_lcb
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > 250.0:
        simple_scores = score_ei
        simple_scores = np.where(np.isfinite(simple_scores), simple_scores, 0.0)
        return simple_scores
    
    return scores
# EVOLVE-BLOCK-END
