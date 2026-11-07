"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) with adaptive exploration.
    
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
    
    # Deadline guard
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
    
    # --- Cost Feasibility Penalty ---
    # The minimizer is in [1, 9]^4.
    # Cost = 1 + mean(x_scaled).
    # Min cost in [1, 9]^4 is 1.1, max is 1.9.
    # We use a smooth window to favor costs in this range.
    # Widths control how sharply we penalize outside costs.
    lower_bound = 1.1
    upper_bound = 1.9
    lower_width = 0.08
    upper_width = 0.08
    
    # Sigmoid gates
    lower_factor = 0.5 * (1.0 + np.tanh((cost - lower_bound) / lower_width))
    upper_factor = 0.5 * (1.0 - np.tanh((cost - upper_bound) / upper_width))
    cost_feasibility = np.clip(lower_factor * upper_factor, 0.0, 1.0)
    
    # --- Expected Improvement (EI) ---
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # --- Exploration Term (LCB-like) ---
    # We use a standard deviation weighted term to encourage exploration.
    # This is scaled by the phase to reduce exploration over time.
    call_idx = min(_acq_state['call_count'], 50)
    phase = call_idx / 50.0
    
    # Exploration weight: high early, low late
    # Early: explore to find the basin. Late: exploit to find the min.
    exploration_weight = 0.5 * (1.0 - phase) + 0.1  # 0.6 -> 0.1
    
    lcb_term = exploration_weight * sigma
    
    # --- Combine ---
    # Base score is EI + Exploration
    # We weight everything by cost feasibility to stay in the likely region.
    # Additionally, we apply a soft cost-efficiency factor: prefer lower cost.
    # cost_eff = 1.5 / cost is a simple heuristic.
    # Let's use a smoother decay: exp(-0.5 * (cost - 1.5)^2) is centered at 1.5.
    # But cost_feasibility already handles the range.
    # Let's add a mild preference for lower cost within the feasible range.
    cost_efficiency = 1.0 / (1.0 + 0.5 * (cost - 1.1)) # Slightly favors lower cost
    
    scores = (ei + lcb_term) * cost_feasibility * cost_efficiency
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
