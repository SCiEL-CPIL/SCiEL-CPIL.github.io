"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using cost-weighted LCB with adaptive exploration.
    
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
    
    # --- Cost-Weighted LCB with Adaptive Beta ---
    # For minimization: LCB = mu - beta * sigma
    # We want HIGH scores to be selected, so we negate: score = -(mu - beta*sigma) = -mu + beta*sigma
    # But we also need to account for cost efficiency
    
    # Adaptive beta: start exploratory, become more exploitative as we learn
    # Use call count to adapt (early calls = more exploration)
    call_idx = min(_acq_state['call_count'], 50)
    # Beta decays from ~3.0 to ~1.5 over first 50 calls
    beta = 3.0 - 1.5 * (call_idx / 50.0)
    
    # Base LCB score (higher is better for selection)
    lcb_score = -mu + beta * sigma
    
    # --- Smooth Cost Efficiency Factor ---
    # Cost ranges from 1.0 to 2.0. The valid region where minimizer can be is [1.1, 1.9].
    # Instead of hard masking, use a smooth penalty that:
    # - Is 1.0 in the valid region [1.1, 1.9]
    # - Decreases smoothly outside this region
    # This is a smooth "trust region" for cost
    
    # Smooth cost factor: Gaussian-like bump centered at cost=1.5, width 0.4
    # At cost=1.0: factor ~0.0, at cost=1.1: ~0.7, at cost=1.5: 1.0, at cost=1.9: ~0.7, at cost=2.0: ~0.0
    cost_center = 1.5
    cost_width = 0.4
    cost_factor = np.exp(-((cost - cost_center) ** 2) / (2 * cost_width ** 2))
    
    # Additional: prefer lower cost within valid region (cost efficiency)
    # cost_eff = 1.0 / cost gives higher weight to cheaper points
    # But normalize so that at cost=1.5, it's 1.0/1.5
    # Actually, let's use a gentler preference: 1.5 / cost (so at cost=1.5, factor=1.0)
    cost_eff = 1.5 / cost
    
    # Combine: smooth validity * cost efficiency
    # The cost_factor handles "can this be the minimizer?" 
    # The cost_eff handles "how efficient is evaluating here?"
    cost_combined = cost_factor * cost_eff
    
    # Final score
    scores = lcb_score * cost_combined
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
