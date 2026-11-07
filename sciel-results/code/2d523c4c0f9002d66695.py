"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Standard EI components
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (minimization perspective)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Global budget-aware scaling strategy:
    # Instead of point-wise division by cost, we use a cost z-score penalty.
    # This penalizes candidates that are more expensive than the average candidate in the batch.
    # This allows for a more global view of the cost landscape relative to the candidate set.
    
    mean_cost = np.mean(cost)
    std_cost = np.std(cost)
    
    # Avoid division by zero if all costs are identical
    if std_cost < 1e-9:
        cost_z = np.zeros_like(cost)
    else:
        cost_z = (cost - mean_cost) / std_cost
    
    # The penalty term is proportional to the cost z-score.
    # We want to subtract a penalty for high cost.
    # The scale of the penalty should be relative to the scale of EI.
    # Let's estimate the scale of EI.
    # If EI is very small, a large penalty might kill all scores.
    # If EI is large, a small penalty might be negligible.
    
    # A robust scale for EI could be the 90th percentile or the mean.
    # Let's use the median of positive EI values to be robust to outliers.
    positive_ei = ei[ei > 1e-8]
    if len(positive_ei) > 0:
        ei_scale = np.median(positive_ei)
        if ei_scale < 1e-8:
            ei_scale = 1.0
    else:
        ei_scale = 1.0
        
    # The penalty coefficient. 
    # We want the penalty to be significant but not overwhelming.
    # A cost z-score of 2 (2 std devs above mean) should incur a penalty.
    # Let's try a penalty of 0.5 * ei_scale for each unit of cost z-score.
    # This means a candidate 2 std devs above mean cost loses 1.0 * ei_scale in score.
    # This seems reasonable.
    
    penalty_coef = 0.5 * ei_scale
    score = ei - penalty_coef * cost_z
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290.0:
        # Fallback to simple EI if taking too long (unlikely)
        return ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(score), score, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
