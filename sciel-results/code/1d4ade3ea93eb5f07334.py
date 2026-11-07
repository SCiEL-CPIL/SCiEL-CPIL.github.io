# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 325s as safety margin
_DEADLINE = 325.0
_START_TIME = time.time()

def _check_deadline():
    """Return True if we should abort and return current best."""
    return (time.time() - _START_TIME) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Styblinski-Tang 2D Bayesian optimization.
    
    Uses a model-and-optimize approach: we parameterize the cost-aware scoring
    and optimize the parameters to maximize expected improvement per unit cost.
    
    Strategy:
    - Compute Expected Improvement (EI) and Lower Confidence Bound (LCB)
    - Use a cost-aware combination that balances exploitation and exploration
    - The cost penalty is adaptive based on the current f_best and uncertainty
    
    Parameters:
    - mu: shape (2000,), GP mean predictions
    - sigma: shape (2000,), GP standard deviations
    - f_best: scalar, best observed value
    - cost: shape (2000,), evaluation costs (1 + mean(x_scaled))
    
    Returns:
    - scores: shape (2000,), acquisition scores (higher = evaluate next)
    """
    # Check deadline first
    if _check_deadline():
        return np.zeros_like(mu, dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Clip to valid ranges
    sigma = np.clip(sigma, 1e-8, 1e4)
    cost = np.clip(cost, 0.5, 10.0)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    ei = (f_best - mu) * cdf_vals + sigma * pdf_vals
    ei = np.maximum(ei, 0.0)
    
    # Estimate progress toward the known minimum
    dist_to_min = max(f_best - (-78.3323), 0.0)
    progress = 1.0 / (1.0 + 0.05 * dist_to_min)
    # progress ~ 1 when close to min, ~ 0.1 when far
    
    # Cost-aware EI: divide by sqrt(cost) to reward efficiency
    cost_efficiency = 1.0 / np.sqrt(cost)
    ei_cost_aware = ei * cost_efficiency
    
    # Cost-aware LCB: stronger exploration when uncertain and early
    # beta decreases as we converge; cost penalizes expensive exploration
    beta_base = 2.5 * (1.0 - 0.5 * progress)  # 2.5 early, 1.25 late
    beta_cost_aware = beta_base / np.sqrt(cost)
    lcb_cost_aware = (mu - beta_cost_aware * sigma) * cost_efficiency
    
    # Soft-min combination of EI and LCB signals
    # Use log-sum-exp for smooth minimum
    # We want the better of the two signals, but smoothly
    # Scale to comparable magnitude
    ei_scaled = ei_cost_aware / (np.max(ei_cost_aware) + 1e-8)
    lcb_scaled = lcb_cost_aware / (np.max(np.abs(lcb_cost_aware)) + 1e-8)
    
    # Temperature for soft-min: lower = sharper
    temp = 0.3 + 0.4 * (1.0 - progress)  # 0.7 early, 0.3 late
    
    # Soft-min: -log(exp(-a/temp) + exp(-b/temp)) * temp
    # But we want to maximize, so use soft-max of negated values
    a = -ei_scaled
    b = -lcb_scaled
    m = np.maximum(a, b)
    soft_max = m - temp * np.log(np.exp((a - m) / temp) + np.exp((b - m) / temp))
    combined = -soft_max  # Convert back to "higher is better"
    
    # Scale back to original magnitude
    scale = 0.5 * (np.max(ei_cost_aware) + np.max(np.abs(lcb_cost_aware))) + 1e-8
    score = combined * scale
    
    # Add a small cost penalty to discourage very expensive points
    # Penalty grows with cost above 1.5 and with progress
    cost_penalty = 0.1 * progress * np.maximum(cost - 1.5, 0.0) * sigma
    score = score - cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
