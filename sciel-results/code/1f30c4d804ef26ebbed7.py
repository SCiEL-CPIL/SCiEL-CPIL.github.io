"""Cost-aware acquisition using Thompson Sampling for Rastrigin-2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # Safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Thompson Sampling.
    
    Instead of deterministic Expected Improvement (EI), we sample from the
    posterior distribution of the objective function at each candidate point.
    We then select the candidate that offers the best expected value per unit
    cost based on this sample. This approach naturally balances exploration
    and exploitation and is robust to multimodality.
    
    Args:
        mu: (2000,) GP mean predictions in objective units.
        sigma: (2000,) GP std predictions in objective units.
        f_best: float, best observed (lowest) value.
        cost: (2000,) evaluation cost per candidate.
    
    Returns:
        scores: (2000,) finite scores, higher = pick next.
    """
    if time.time() > _DEADLINE:
        # Fallback: deterministic cost-aware heuristic
        # Prefer cheap points, slightly penalized by uncertainty if possible,
        # but 1/cost is a safe default.
        return 1.0 / (cost + 1e-9)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample from the posterior: y ~ N(mu, sigma^2)
    # Using standard normal random numbers
    samples = mu + sigma_safe * np.random.randn(len(mu))
    
    # We want to minimize the objective function.
    # The cost-aware score should favor low sampled values and low costs.
    # Score = -sample / cost
    # This is equivalent to maximizing (sample / cost) with a negative sign,
    # or minimizing (sample / cost).
    # Since we want higher scores to be better, we return -sample / cost.
    
    scores = -samples / (cost + 1e-9)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
