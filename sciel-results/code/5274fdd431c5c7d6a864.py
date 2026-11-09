"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 325.0  # Safety margin before the 330s kill limit
_eval_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining raw Expected Improvement with an adaptive
    exploration bonus that decreases as evaluations progress.
    
    The exploration bonus is capped at a fraction of the median sigma to
    prevent pure exploration, and its weight smoothly transitions from
    high (early) to low (late) based on evaluation count.
    
    Args:
        mu: Array of predicted means (shape (2000,)).
        sigma: Array of predicted standard deviations (shape (2000,)).
        f_best: The best observed value so far.
        
    Returns:
        Array of scores (shape (2000,)), higher is better.
    """
    global _eval_count
    
    # Check for timeout
    if time.time() - _start_time > _TIMEOUT:
        return np.zeros_like(mu)
    
    _eval_count += 1
    
    # Ensure sigma is non-negative and finite, avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # --- Expected Improvement (primary exploitation signal) ---
    z = (f_best - mu) / sigma
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # --- Adaptive Exploration Bonus ---
    # Reference scale: median sigma across candidates
    sigma_ref = np.median(sigma)
    
    # Exploration term: how much more uncertain is this point vs median?
    # Cap it so that no point gets more than sigma_ref bonus
    raw_exploration = np.maximum(sigma - sigma_ref, 0.0)
    capped_exploration = np.minimum(raw_exploration, sigma_ref)
    
    # Dynamic weight: starts high (exploration-heavy) and decays to near-zero
    # Total budget is 30 evaluations (5 initial + 25 more)
    # We've already used _eval_count evaluations in the acquisition loop
    total_budget = 30
    progress = min(_eval_count / total_budget, 1.0)
    
    # Exploration weight: from 0.15 (early) to 0.02 (late)
    # This keeps EI dominant but allows meaningful exploration early
    w_explore = 0.15 - 0.13 * progress
    
    scores = ei + w_explore * capped_exploration
    
    # Clean up non-finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
