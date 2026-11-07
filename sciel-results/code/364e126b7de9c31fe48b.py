"""
Bayesian Optimization Acquisition Function for Hartmann 6d
Strategy: Deterministic LCB with adaptive exploration to balance
global coverage in multimodal landscapes within a tight 30-eval budget.
"""
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses an adaptive Lower Confidence Bound (LCB) to balance exploration
    and exploitation, tuned for the 6D Hartmann function with 30 evaluations.
    
    Parameters:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - best observed value (minimum)
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    start_time = time.time()
    deadline = 300.0  # Safety margin before the 330s kill limit

    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)

    # Adaptive LCB: use a fixed but aggressive exploration coefficient.
    # For Hartmann 6D with 30 evaluations, we need strong exploration early
    # but the budget is too small for many adaptive schemes.
    # kappa=4.0 encourages even more aggressive exploration of high-uncertainty
    # regions, aiming to escape local optima more effectively than kappa=3.0.
    # This is based on the observation that LCB (3.0) outperformed EI, suggesting
    # that further exploration bias is beneficial.
    kappa = 4.0

    # LCB score: -mu + kappa * sigma
    # Higher score = more promising (lower expected value, higher uncertainty)
    scores = -mu + kappa * sigma_safe

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)

    # Check deadline
    if time.time() - start_time > deadline:
        return scores

    return scores
