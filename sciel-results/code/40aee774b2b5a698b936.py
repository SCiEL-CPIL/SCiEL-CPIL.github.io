"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer to enforce the 330s wall-clock limit with a safety margin
_start_time = time.time()
TIME_LIMIT = 320.0 # Leave 10s buffer for cleanup/return

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI) and 
    Lower Confidence Bound (LCB) with adaptive exploration.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Check time limit
    if time.time() - _start_time > TIME_LIMIT:
        return np.zeros_like(mu)
    
    # Ensure finite inputs and proper types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    f_best_val = float(f_best)
    
    # Calculate z-score for EI
    z = (f_best_val - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # E[I(x)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    ei = (f_best_val - mu) * cdf_val + sigma_safe * pdf_val
    ei = np.maximum(ei, 0.0)
    
    # LCB score (higher is better for our purpose)
    # Adaptive beta: start moderate, reduce as f_best approaches 0
    # This balances exploration early and exploitation later
    beta = 1.5 * np.exp(-0.5 * np.abs(f_best_val))
    
    lcb_score = -(mu - beta * sigma_safe)
    
    # Combine EI and LCB with adaptive weighting
    # As f_best gets closer to 0 (better), we weight EI more heavily
    # Early on (f_best large), we weight LCB more for exploration
    ei_weight = 1.0 / (1.0 + np.abs(f_best_val))
    lcb_weight = 1.0 - ei_weight
    
    # Normalize EI and LCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    lcb_max = np.max(np.abs(lcb_score)) if np.max(np.abs(lcb_score)) > 0 else 1.0
    
    ei_normalized = ei / ei_max
    lcb_normalized = lcb_score / lcb_max
    
    # Weighted combination
    combined = ei_weight * ei_normalized + lcb_weight * lcb_normalized
    
    # Cost-aware weighting: prefer lower cost points
    # cost ranges from ~1 to ~2, so 1/cost ranges from ~0.5 to ~1.0
    cost_weight = 1.0 / np.maximum(cost, 1e-9)
    
    # Final score
    scores = combined * cost_weight
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
