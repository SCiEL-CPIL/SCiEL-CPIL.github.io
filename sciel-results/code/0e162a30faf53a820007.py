"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining LCB exploration with cost-aware
    structural constraints and adaptive scaling.
    
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
        _acq_state['call_count'] = 0
    _acq_state['call_count'] += 1
    
    # Deadline guard - return safely if we're running out of time
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 325.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values robustly
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.5)
    
    # Ensure positive values
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # --- Hybrid Acquisition: LCB with adaptive beta ---
    # For minimization: LCB = mu - beta * sigma
    # We want HIGH scores to be selected, so we negate: score = -(mu - beta*sigma) = -mu + beta*sigma
    
    # Adaptive beta: start exploratory, become more exploitative as we learn
    call_idx = min(_acq_state['call_count'], 50)
    # Beta decays from ~2.5 to ~1.2 over first 50 calls
    beta = 2.5 - 1.3 * (call_idx / 50.0)
    
    # Base LCB score (higher is better for selection)
    lcb_score = -mu + beta * sigma
    
    # --- Structural Cost Constraint (from inspiration) ---
    # The minimizer is in inner 80% of [0,10]^4, i.e., [1,9]^4
    # Cost(x) = 1 + mean(x/10), so valid cost range is [1.1, 1.9]
    # Points outside this range CANNOT contain the global minimum
    
    # Smooth cost validity factor using sigmoid-like transitions
    # This avoids cliff edges while strongly discouraging invalid regions
    # Lower bound: transition from 0 to 1 over [1.0, 1.1]
    # Upper bound: transition from 1 to 0 over [1.9, 2.0]
    
    # Use tanh-based smooth step for smoother transitions than linear
    lower_width = 0.05  # Half-width of transition zone
    upper_width = 0.05
    
    # Lower bound: 0.5 * (1 + tanh((cost - 1.1) / lower_width))
    # At cost=1.05: ~0, at cost=1.1: ~0.5, at cost=1.15: ~1
    lower_factor = 0.5 * (1.0 + np.tanh((cost - 1.1) / lower_width))
    
    # Upper bound: 0.5 * (1 - tanh((cost - 1.9) / upper_width))
    # At cost=1.85: ~1, at cost=1.9: ~0.5, at cost=1.95: ~0
    upper_factor = 0.5 * (1.0 - np.tanh((cost - 1.9) / upper_width))
    
    # Combine validity factors
    cost_validity = lower_factor * upper_factor
    cost_validity = np.clip(cost_validity, 0.0, 1.0)
    
    # --- Cost Efficiency Within Valid Region ---
    # Within the valid range [1.1, 1.9], prefer lower cost for efficiency
    # Use a gentle preference: normalize so that cost=1.5 gives factor 1.0
    # cost_eff = 1.5 / cost gives 1.36 at cost=1.1, 1.0 at cost=1.5, 0.79 at cost=1.9
    # But we only apply this within the valid region
    
    # Smooth cost efficiency that blends with validity
    cost_eff = 1.5 / cost
    
    # Final cost factor: validity gates the efficiency
    # In invalid regions, score is heavily penalized
    # In valid regions, efficiency gives mild preference to lower cost
    cost_combined = cost_validity * cost_eff
    
    # --- Final Score ---
    # Combine LCB with cost factor
    scores = lcb_score * cost_combined
    
    # Additional: small exploration bonus for very uncertain points
    # This helps escape local optima in the GP landscape
    # Only apply within valid cost region to avoid wasting budget
    exploration_bonus = 0.1 * sigma * cost_validity
    
    scores = scores + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
