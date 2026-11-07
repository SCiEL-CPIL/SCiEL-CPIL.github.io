"""
Cost-aware acquisition function for Bayesian optimization (shekel 4d).
Implements a combination of Cost-Aware EI and an exploration bonus.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # Seconds to spare before the hard limit

def _get_time_remaining():
    """
    Calculate time remaining before the hard 300s limit per evaluation,
    or the global 330s limit if we are tracking total wall clock.
    The contract says: 'Each evaluation must finish within 300 s'
    and 'evaluator KILLS the program after 330s wall-clock'.
    We should be safe against both.
    """
    elapsed = time.time() - _start_time
    # We don't know the exact start of the current evaluation vs previous ones easily 
    # without passing state, but _start_time is the module load time.
    # To be safe, we assume the "300s per evaluation" is the stricter constraint 
    # that we must respect individually, but since we can't measure per-call duration 
    # inside the function without overhead, we rely on the fact that the function 
    # is fast. 
    # However, the prompt says "Keep an EXPLICIT deadline guard: track elapsed time and 
    # return the best result found with a safety margin".
    # This implies we should check if the TOTAL time is getting close to 330s.
    
    total_elapsed = time.time() - _start_time
    return 330.0 - total_elapsed

def acquisition(mu, sigma, f_best, cost):
    """
    Compute cost-aware acquisition scores.
    
    Args:
        mu: Mean of the GP prediction (shape: 2000,)
        sigma: Std deviation of the GP prediction (shape: 2000,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each point (shape: 2000,)
        
    Returns:
        scores: Acquisition scores (shape: 2000,), higher is better.
    """
    # Deadline Guard
    # If we are very close to the 330s global kill switch, use the fastest possible method
    if _get_time_remaining() < _DEADLINE_MARGIN:
        # Fast fallback: simple EI / cost
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        # Approximate EI for speed if needed, but norm.cdf is fast enough for 2000 pts
        # However, to be absolutely safe on time, we can use a rough approximation or just return zeros?
        # No, we must return finite scores. Let's stick to the standard fast path.
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # --- Main Algorithm ---
    
    # Ensure sigma and cost are safe for division
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # 1. Standard Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    # Clamp to non-negative to handle numerical artifacts
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI
    cea = ei / cost_safe
    
    # 2. Exploration Bonus
    # We want to explore areas where the model is uncertain (high sigma) 
    # or where the cost is low, even if the immediate EI is low.
    # A linear surrogate (UCB-style) works well here:
    # ucb = (f_best - mu + beta * sigma) / cost
    # We use a small beta to balance exploitation (CEA) and exploration (UCB).
    # Beta is tuned for the Shekel function scale.
    beta = 1.5
    
    ucb_score = (f_best - mu + beta * sigma_safe) / cost_safe
    ucb_score = np.maximum(ucb_score, 0.0)
    
    # Combine terms.
    # CEA drives exploitation. UCB drives exploration of uncertain/cheap regions.
    # The weight on UCB is chosen to be significant but not dominant.
    w_explore = 0.2
    
    scores = cea + w_explore * ucb_score
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
