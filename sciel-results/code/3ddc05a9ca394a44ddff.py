"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    This function computes a stochastic acquisition score by sampling from the GP posterior
    and evaluating the cost-adjusted sample value. Higher scores indicate better candidates.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - lowest observed function value
        cost: array of shape (2000,) - cost of evaluating each candidate
        
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a valid finite array to avoid invalidating the result
        return np.random.uniform(size=mu.shape)
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling: Draw a single random sample from the GP posterior
    # sample(x) = mu(x) + sigma(x) * z, where z ~ N(0, 1)
    z = np.random.standard_normal(mu.shape)
    sample = mu + sigma_safe * z
    
    # Cost-aware scoring: We want to minimize the sampled value per unit cost.
    # Since higher scores are better for selection, we invert the cost-adjusted value.
    # A common approach is to use -sample / cost or -sample * (1/cost).
    # However, to ensure the score is meaningful and finite, and to align with
    # the "higher is better" requirement, we can use:
    # score = - (sample / cost)
    # This prioritizes low sampled values and low costs.
    
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute the cost-adjusted sampled value
    adjusted_sample = sample / cost_safe
    
    # The acquisition score is the negative of the adjusted sample, so that
    # lower sampled values (and lower costs) yield higher scores.
    scores = -adjusted_sample
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
