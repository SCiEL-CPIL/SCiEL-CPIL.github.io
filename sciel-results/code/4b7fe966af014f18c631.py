"""Rewritten acquisition for Bayesian optimization of Hartmann-3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) with an Adaptive Uncertainty Bonus.
    
    This approach uses EI as the primary exploitation driver, which has proven effective 
    for this task. To address the multimodal nature of Hartmann-3D and the small budget, 
    we add an exploration bonus proportional to the posterior standard deviation. 
    Crucially, the weight of this bonus is adapted based on the global uncertainty 
    (mean sigma). This encourages stronger exploration in the early stages when the 
    posterior is uncertain (high sigma) and shifts towards pure exploitation (EI) 
    as the model becomes more certain (low sigma).
    
    Parameters:
    mu: array of shape (n,) - posterior means
    sigma: array of shape (n,) - posterior standard deviations
    f_best: float - best observed value (lowest)
    
    Returns:
    scores: array of shape (n,) - finite scores, higher means evaluate next
    """
    _ensure_start_time()
    
    # Safety guard: if we're getting close to the 300s limit, return a fast fallback
    elapsed = time.time() - _START_TIME
    if elapsed > 295:
        sigma_safe = np.maximum(sigma, 1e-10)
        # Fallback: Simple UCB-like score with fixed small beta
        scores = -mu + 0.1 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is non-negative and safe for operations
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive Exploration Weight
    # Estimate global uncertainty. Since targets are standardized, initial sigma ~ 1.0.
    # As BO progresses, sigma decreases.
    mean_sigma = np.mean(sigma_safe)
    
    # We want the exploration bonus to be significant early on.
    # Let's scale the weight by the mean sigma.
    # Base weight 0.5 means if mean_sigma=1, beta=0.5.
    # If mean_sigma=0.1, beta=0.05.
    # This effectively decays exploration as the model learns.
    exploration_weight = 0.5 * mean_sigma
    
    # Add the adaptive exploration bonus
    ucb_component = exploration_weight * sigma_safe
    
    scores = ei + ucb_component
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
