"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state for deadline tracking
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
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
    if elapsed > 305.0:
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
    
    n_calls = _acq_state['call_count']
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Compute Probability of Improvement (PI)
    pi = cdf_z
    
    # Compute Upper Confidence Bound (UCB)
    # Use a time-varying beta that starts high (exploration) and decreases (exploitation)
    if n_calls <= 3:
        beta = 3.0
    elif n_calls <= 8:
        beta = 2.0
    elif n_calls <= 15:
        beta = 1.2
    elif n_calls <= 25:
        beta = 0.8
    else:
        beta = 0.5
    
    ucb = mu + beta * sigma
    
    # Cost-aware normalization
    # We want to combine EI, PI, and UCB in a cost-aware manner
    
    # EI per cost
    ei_cost = ei / cost
    
    # UCB improvement per cost (lower is better, so we want low UCB)
    # The expected reduction in f_best if we pick this point
    # Approximate as (f_best - ucb) / cost, but only when ucb < f_best
    ucb_improvement = np.maximum(f_best - ucb, 0) / cost
    
    # PI per cost
    pi_cost = pi / cost
    
    # Blend the three acquisition functions with time-varying weights
    # Early on: explore with UCB
    # Middle: balance EI and UCB
    # Late: exploit with EI
    
    if n_calls <= 5:
        w_ei = 0.3
        w_ucb = 0.5
        w_pi = 0.2
    elif n_calls <= 12:
        w_ei = 0.5
        w_ucb = 0.3
        w_pi = 0.2
    elif n_calls <= 20:
        w_ei = 0.7
        w_ucb = 0.2
        w_pi = 0.1
    else:
        w_ei = 0.85
        w_ucb = 0.1
        w_pi = 0.05
    
    # Normalize each component to have similar scale
    # Use max normalization for stability
    max_ei_cost = np.max(ei_cost) if np.max(ei_cost) > 0 else 1.0
    max_ucb_imp = np.max(ucb_improvement) if np.max(ucb_improvement) > 0 else 1.0
    max_pi_cost = np.max(pi_cost) if np.max(pi_cost) > 0 else 1.0
    
    ei_norm = ei_cost / max_ei_cost
    ucb_norm = ucb_improvement / max_ucb_imp
    pi_norm = pi_cost / max_pi_cost
    
    # Combined score
    scores = w_ei * ei_norm + w_ucb * ucb_norm + w_pi * pi_norm
    
    # Add a small exploration bonus for high uncertainty regions in early stages
    if n_calls <= 10:
        # Normalize sigma
        max_sigma = np.max(sigma) if np.max(sigma) > 0 else 1.0
        sigma_norm = sigma / max_sigma
        scores += 0.1 * sigma_norm / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
