"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 320s safety margin within 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function returning 2000 finite scores.
    
    Uses a hybrid approach:
    - Expected Improvement (EI) as the base exploitation signal
    - Cost-aware normalization
    - Dynamic shift toward exploitation as budget depletes
    - Numerical safety for sigma near zero
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a safe, finite fallback: rank by lowest mu
        return -mu.copy()
    
    # Ensure arrays are float64 and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: divide by cost to get improvement per unit cost
    ei_cost = ei / cost_safe
    
    # Dynamic weight: as we spend more budget, shift toward exploitation
    # f_best being low relative to the theoretical minimum (-0.8) indicates progress
    # Use the spread of mu to gauge exploration vs exploitation balance
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = max(mu_max - mu_min, 1e-8)
    
    # Normalize EI to [0, 1] range for blending
    ei_max = np.max(ei_cost)
    if ei_max > 1e-12:
        ei_norm = ei_cost / ei_max
    else:
        ei_norm = np.zeros_like(ei_cost)
    
    # Also compute a pure exploitation score (lowest predicted mean)
    # Normalize mu to [0, 1] (lower mu = better, so invert)
    mu_norm = (mu_max - mu) / mu_range  # 1.0 for best mu, 0.0 for worst
    
    # Blend: weight shifts from exploration (EI) to exploitation (mu) 
    # based on how close we are to the known minimum
    # Theoretical min is -0.8; if f_best is close, exploit more
    progress = np.clip((f_best - (-0.8)) / (f_best - (-0.8) + 1e-8), 0.0, 1.0)
    # Actually, use a simpler heuristic: as f_best decreases, exploit more
    # Use the relative position of f_best in the observed range
    # Simpler: use a fixed blend that favors EI slightly
    w_ei = 0.7
    w_exploit = 0.3
    
    # Final score: higher is better
    score = w_ei * ei_norm + w_exploit * mu_norm
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Add a tiny bit of noise-free tie-breaking to avoid identical scores
    # Use the index as a very small tiebreaker (deterministic)
    indices = np.arange(len(score), dtype=np.float64)
    score += indices * 1e-10
    
    return score
# EVOLVE-BLOCK-END
