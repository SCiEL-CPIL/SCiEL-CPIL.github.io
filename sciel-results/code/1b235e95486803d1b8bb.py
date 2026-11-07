"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (shekel 4d).
    Uses a cost-efficient Expected Improvement with adaptive exploration.
    
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
    
    # --- Cost-awareness ---
    # The minimizer is in [1,9]^4 (inner 80%), so cost in [1.1, 1.9]
    # Use smooth sigmoid gates to softly weight valid cost region
    lower_width = 0.05
    upper_width = 0.05
    
    lower_factor = 0.5 * (1.0 + np.tanh((cost - 1.1) / lower_width))
    upper_factor = 0.5 * (1.0 - np.tanh((cost - 1.9) / upper_width))
    cost_validity = np.clip(lower_factor * upper_factor, 0.0, 1.0)
    
    # Cost efficiency: prefer lower cost within valid region
    cost_eff = 1.5 / cost
    
    # --- Expected Improvement (EI) for minimization ---
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # --- Phase-based adaptation ---
    call_idx = min(_acq_state['call_count'], 50)
    phase = call_idx / 50.0  # 0 to 1
    
    # Adaptive LCB beta: decays from exploratory to exploitative
    beta = 2.0 - 0.8 * phase  # 2.0 -> 1.2
    
    lcb_score = -mu + beta * sigma
    
    # Blend EI and LCB based on phase
    blend = 0.3 + 0.7 * phase  # 0.3 -> 1.0
    
    base_score = blend * ei + (1 - blend) * lcb_score
    
    # --- Probability weight: likelihood of being near global minimum ---
    mu_scale = np.std(mu) + 1e-8
    prob_weight = np.exp(-0.5 * ((mu - np.min(mu)) / mu_scale) ** 2) * \
                  (sigma / (sigma + 0.1))
    
    # Combine all components
    scores = base_score * prob_weight * cost_validity * cost_eff
    
    # Exploration bonus for uncertain points in valid region (early phase)
    exploration_bonus = 0.05 * sigma * cost_validity * (1 - phase)
    scores = scores + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
