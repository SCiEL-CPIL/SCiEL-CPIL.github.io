# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Known minimum of Hartmann 6d
HARTMANN_6D_MIN = -3.32237

# Global state for adaptive acquisition
_acq_state = {
    'elapsed': 0.0,
    'call_count': 0,
    'best_score': None,
    'last_mu': None,
    'last_sigma': None,
    'last_f_best': None,
    'phase': 'explore',  # 'explore' or 'exploit'
    'explore_weight': 1.0,
}


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6d.
    Uses a hybrid acquisition that balances exploration and exploitation
    with adaptive weighting based on progress.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array of shape (2000,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        The best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores, higher values indicate more promising points.
    """
    global _acq_state
    
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace any non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-3)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Track call count and elapsed time
    _acq_state['call_count'] += 1
    current_time = time.time()
    if _acq_state['elapsed'] == 0.0:
        _acq_state['elapsed'] = current_time
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Upper Confidence Bound (UCB)
    # Use a temperature parameter that adapts over time
    t = _acq_state['call_count']
    # Start with more exploration, shift to exploitation
    # Use a schedule that reduces exploration over time
    if t < 10:
        beta = 2.5  # High exploration early
    elif t < 20:
        beta = 1.5  # Moderate exploration
    else:
        beta = 0.8   # More exploitation later
    
    ucb = mu + beta * sigma
    
    # Combine EI and UCB with adaptive weighting
    # Early: more UCB (exploration), Later: more EI (exploitation)
    if t < 15:
        w_ei = 0.3
        w_ucb = 0.7
    elif t < 25:
        w_ei = 0.5
        w_ucb = 0.5
    else:
        w_ei = 0.7
        w_ucb = 0.3
    
    # Normalize scores to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(-ucb) if np.max(-ucb) > 0 else 1.0  # We want to minimize ucb
    
    # For UCB in minimization context, higher ucb means more uncertain
    # We want to balance: low mu (good) and high sigma (uncertain)
    # UCB for minimization: score = -mu + beta * sigma (higher is better for exploration)
    ucb_score = -mu + beta * sigma
    
    # Normalize both to [0, 1] range for combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb_score), np.max(ucb_score)
    
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_max > ucb_min:
        ucb_norm = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Combined score
    combined = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    # Scale back to a reasonable range
    combined = combined * (ei_max if ei_max > 0 else 1.0)
    
    return combined
# EVOLVE-BLOCK-END
