"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds


def _is_deadline_exceeded():
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Thompson Sampling with cost normalization.
    
    Instead of computing Expected Improvement (a deterministic improvement-based
    criterion), we sample from the posterior at each candidate and select the
    one with the best predicted value per unit cost. This provides natural
    exploration-exploitation balance through the sampling process.
    """
    # Deadline guard: if we're close to the limit, return a simple fallback
    if _is_deadline_exceeded():
        # Return scores that prioritize low cost and low predicted mean
        return -(mu + 0.1 * cost)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost normalization
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample from the posterior N(mu, sigma)
    # Use a deterministic seed based on f_best for reproducibility within a run
    # but different across iterations as f_best changes
    rng = np.random.default_rng(int(f_best * 1e6) % (2**32))
    
    # Sample from the posterior for each candidate
    # The sampled value represents a plausible true function value
    sample = mu + sigma_safe * rng.standard_normal(len(mu))
    
    # Score: lower sampled value is better, normalized by cost
    # We want to maximize (higher = evaluate next), so negate
    score = -sample / cost_safe
    
    # Add a small uncertainty bonus to encourage exploration in uncertain regions
    # This helps when the GP is poorly calibrated
    uncertainty_bonus = 0.02 * sigma_safe / cost_safe
    
    final_score = score + uncertainty_bonus
    
    # Ensure finite values
    final_score = np.where(np.isfinite(final_score), final_score, 0.0)
    
    return final_score
# EVOLVE-BLOCK-END
