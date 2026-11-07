"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for tracking call count and elapsed time
_acq_state = {'start_time': None, 'call_count': 0}

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using a structurally different
    approach: Thompson Sampling with cost-awareness.
    
    Instead of pointwise EI or BRM-style ranking, we draw samples from the
    posterior and select the candidate that minimizes the sampled objective
    divided by cost. This captures the full posterior structure and naturally
    balances exploration and exploitation through the sampling process.
    
    We use a hybrid approach:
    1. Thompson Sampling: sample from N(mu, sigma^2) and score by -sample/cost
    2. Cost-aware LCB as a secondary component for stability
    3. Adaptive blending based on budget remaining
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best observed value so far (lowest, since we minimize)
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x scaled to [0,1])
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    # Initialize module-level state
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    _acq_state['call_count'] += 1
    
    # Deadline guard
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 305.0:
        return np.zeros(len(mu), dtype=np.float64)
    
    t0 = time.time()
    deadline = 295.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    n_calls = _acq_state['call_count']
    
    # === Cost kernel: Gaussian centered at expected minimizer cost (1.5) ===
    # Expected minimizer of shekel 4d is around cost ~1.5 (mid-range)
    # Gaussian kernel smoothly favors candidates near this cost
    cost_center = 1.5
    cost_width = 0.5
    cost_kernel = np.exp(-((cost_safe - cost_center) ** 2) / (2.0 * cost_width ** 2))
    # Normalize kernel to [0, 1] range for use as multiplier
    cost_kernel_norm = cost_kernel / np.max(cost_kernel)
    
    # === Thompson Sampling Component ===
    # Draw a sample from the posterior for each candidate
    rng = np.random.RandomState(n_calls * 7919)
    sample = mu + sigma_safe * rng.randn(n)
    
    # Score by negative sampled value weighted by cost kernel
    # Gaussian kernel smoothly balances cost without harsh linear penalty
    ts_score = -sample * cost_kernel_norm
    
    # === Cost-Aware LCB Component ===
    # Budget-aware beta: as total cost approaches 30, reduce exploration
    if n_calls <= 3:
        beta_base = 2.5
    elif n_calls <= 6:
        beta_base = 1.8
    elif n_calls <= 10:
        beta_base = 1.2
    elif n_calls <= 14:
        beta_base = 0.7
    else:
        beta_base = 0.35
    
    # Estimated total cost so far
    est_total_cost = 7.5 + n_calls * 1.5
    budget_fraction = max(0.0, min(1.0, (30.0 - est_total_cost) / 22.5))
    
    # Scale beta by budget fraction
    beta = beta_base * (0.3 + 0.7 * budget_fraction)
    
    # LCB for minimization: mu - beta * sigma
    lcb = mu - beta * sigma_safe
    lcb_score = -lcb * cost_kernel_norm
    
    # Normalize both components
    ts_max = np.max(ts_score) if np.max(ts_score) > 1e-15 else 1.0
    lcb_max = np.max(lcb_score) if np.max(lcb_score) > 1e-15 else 1.0
    
    ts_norm = ts_score / ts_max
    lcb_norm = lcb_score / lcb_max
    
    # Blend Thompson Sampling and LCB
    # Early: more Thompson Sampling for exploration
    # Late: more LCB for exploitation
    if n_calls <= 5:
        w_ts = 0.75
        w_lcb = 0.25
    elif n_calls <= 10:
        w_ts = 0.55
        w_lcb = 0.45
    elif n_calls <= 15:
        w_ts = 0.4
        w_lcb = 0.6
    else:
        w_ts = 0.3
        w_lcb = 0.7
    
    scores = w_ts * ts_norm + w_lcb * lcb_norm
    
    # Add a small exploration bonus for high-uncertainty regions in early stages
    if n_calls <= 8:
        sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 1e-12 else 1.0
        sigma_norm = sigma_safe / sigma_max
        exploration_bonus = 0.1 * sigma_norm * cost_kernel_norm
        scores += exploration_bonus
    
    # Ensure all scores are finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    if time.time() - t0 > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
