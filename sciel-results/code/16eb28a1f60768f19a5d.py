"""Scale-invariant cost-aware UCB acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Probability of Improvement with cost efficiency.
    
    Structural change: Uses the probability of improvement P(f(x) < f_best | GP)
    as the primary acquisition signal, combined with a cost-efficiency bonus.
    This is a different modeling family from UCB: instead of a deterministic
    upper bound, we use the actual probability mass below f_best, which naturally
    adapts to the GP's uncertainty structure.
    
    mu, sigma, f_best, cost: each shape (2000,)
    f_best: scalar (lowest observed value)
    cost: per-candidate cost = 1 + mean(x_scaled)
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative values
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute probability of improvement: P(f(x) < f_best | mu, sigma)
    # For a normal distribution: P(X < f_best) = Phi((f_best - mu) / sigma)
    # We need to handle sigma ~ 0 carefully
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # z-score for probability of improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) using the error function for stability
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    from scipy.special import erf
    poi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    
    # Clip to valid probability range
    poi = np.clip(poi, 1e-15, 1.0 - 1e-15)
    
    # Cost efficiency: reward candidates with lower cost
    # Use log(cost) to make it more sensitive to cost differences
    cost_mean = np.mean(cost_safe)
    cost_efficiency = np.log(cost_mean / cost_safe)
    # Normalize cost efficiency to be roughly in [0, 1] range
    cost_eff_min = np.min(cost_efficiency)
    cost_eff_max = np.max(cost_efficiency)
    cost_eff_range = cost_eff_max - cost_eff_min
    if cost_eff_range < 1e-9:
        cost_eff_normalized = 0.5 * np.ones_like(cost_efficiency)
    else:
        cost_eff_normalized = (cost_efficiency - cost_eff_min) / cost_eff_range
    
    # Uncertainty bonus: reward high sigma (exploration)
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean < 1e-9:
        uncertainty_bonus = np.zeros_like(sigma_safe)
    else:
        uncertainty_bonus = sigma_safe / sigma_mean
    
    # Combine: POI is primary, with cost efficiency and uncertainty as modifiers
    # The POI term dominates; cost and uncertainty modulate it
    score = poi * (1.0 + 0.3 * cost_eff_normalized + 0.2 * uncertainty_bonus)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
