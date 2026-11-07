"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking and per-instance counters
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Combines EI-based exploitation with adaptive LCB exploration and
    smooth Gaussian cost gating.
    
    Parameters:
    - mu: array of shape (2000,), GP mean predictions in objective units
    - sigma: array of shape (2000,), GP standard deviations in objective units
    - f_best: scalar, lowest observed function value
    - cost: array of shape (2000,), evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,), acquisition values (higher = evaluate next)
    """
    # Initialize module-level state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    _acq_state['call_count'] += 1
    
    # Deadline guard: if we're getting close to the 330s wall-clock limit,
    # return a safe default to avoid timeout.
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 308.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    # Ensure proper array types and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Ensure sigma is at least a tiny positive value
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Adaptive LCB: balance exploration/exploitation via a beta parameter
    # that shifts from exploration-heavy to exploitation-heavy over time
    n_calls = _acq_state['call_count']
    # Estimate remaining evaluations: budget 30, each eval costs ~1.5 avg
    # So ~20 total evals possible. We've done n_calls.
    remaining_frac = max(0.0, 1.0 - n_calls / 20.0)
    # Beta: early=2.5 (explore), late=0.5 (exploit)
    beta = 0.5 + 2.0 * remaining_frac
    lcb = mu - beta * sigma
    # Higher LCB score = lower predicted value with uncertainty bonus
    lcb_score = -lcb  # negate so higher is better
    
    # Smooth Gaussian cost gating:
    # Cost is 1 + mean(x scaled to [0,1]), so valid costs are in [1, 2]
    # We want to strongly prefer costs in the middle range [1.1, 1.9]
    # and smoothly penalize extremes
    cost_center = 1.5
    cost_width = 0.4
    cost_gate = np.exp(-0.5 * ((cost - cost_center) / cost_width) ** 2)
    # Normalize cost gate to [0, 1]
    cost_gate = cost_gate / (np.max(cost_gate) + 1e-10)
    
    # Cost efficiency: EI per unit cost (exploitation component)
    ei_cost = ei / cost
    
    # LCB per unit cost (exploration component)
    lcb_cost = lcb_score / cost
    
    # Scale EI and LCB to comparable ranges
    ei_max = np.max(ei_cost) + 1e-10
    lcb_max = np.max(lcb_cost) + 1e-10
    ei_norm = ei_cost / ei_max
    lcb_norm = lcb_cost / lcb_max
    
    # Blend EI and LCB based on phase:
    # Early: more LCB (exploration), Late: more EI (exploitation)
    blend = 0.3 + 0.7 * (1.0 - remaining_frac)  # 0.3 early, 1.0 late
    combined = blend * ei_norm + (1.0 - blend) * lcb_norm
    
    # Apply smooth cost gate
    scores = combined * (0.5 + 0.5 * cost_gate)
    
    # Add a small cost-efficiency bonus for below-average cost
    cost_mean = np.mean(cost)
    cost_std = np.std(cost) + 1e-10
    cost_z = (cost - cost_mean) / cost_std
    cost_bonus = 0.05 * (-cost_z)
    cost_bonus = np.clip(cost_bonus, -0.1, 0.1)
    
    scores = scores * (1.0 + cost_bonus)
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
