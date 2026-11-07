import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time and step count
_acq_state = {
    'start_time': None,
    'step': 0
}

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores for the given mean and standard deviation.
    
    Parameters:
    mu: np.ndarray of shape (N,) - Posterior mean
    sigma: np.ndarray of shape (N,) - Posterior standard deviation
    f_best: float - Best observed value so far (minimum)
    
    Returns:
    np.ndarray of shape (N,) - Acquisition scores (higher is better)
    """
    
    # Initialize state if not present
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    
    # Deadline guard: if we're close to the limit, return a safe fallback
    # The limit is 330s. We want to finish well before that.
    # Since this function is fast, this is mostly a safety net.
    elapsed = time.time() - _acq_state['start_time']
    if elapsed > 300:
         # If we are late, just return something simple to avoid hanging
         # But usually this won't happen.
         # Returning a constant score ensures termination if the loop is stuck.
         return np.zeros_like(mu)
    
    _acq_state['step'] += 1
    step = _acq_state['step']
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei_score = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # 2. Compute UCB score
    # Dynamic beta: start high for exploration, decrease for exploitation
    # In 6D, we need more exploration initially.
    # Total steps = 25 (30 evals - 5 init)
    # Linear decay from 2.0 to 0.5
    beta = 2.0 - (1.5 * step / 25.0)
    ucb_score = f_best - mu + beta * sigma_safe
    
    # 3. Combine scores
    # We want to balance exploration (UCB) and exploitation (EI).
    # Early on, favor UCB. Later, favor EI.
    # w_ucb starts at 0.8, ends at 0.3
    w_ucb = 0.8 - (0.5 * step / 25.0)
    w_ei = 1.0 - w_ucb
    
    scores = w_ucb * ucb_score + w_ei * ei_score
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
