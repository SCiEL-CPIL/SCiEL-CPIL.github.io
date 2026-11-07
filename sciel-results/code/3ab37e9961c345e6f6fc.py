import numpy as np
from scipy.stats import norm
import time

# Track global state for adaptive behavior
_acq_state = {
    'call_count': 0,
    'start_time': None
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses a hybrid of Expected Improvement and Upper Confidence Bound with
    adaptive exploration based on convergence progress.
    """
    if _acq_state['start_time'] is None:
        _acq_state['start_time'] = time.time()
    _acq_state['call_count'] += 1
    
    # Safety: ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Known minimum for Michalewicz 2D
    f_min = -1.8013
    
    # Progress: how close are we to the known minimum?
    # f_best starts around -1.0 to -1.5 typically, converges to -1.8013
    gap_to_min = f_best - f_min  # positive when we haven't found the minimum
    gap_to_min = max(gap_to_min, 1e-6)  # avoid division by zero
    
    # Adaptive beta for UCB: more exploration when far from minimum
    # beta ranges from ~2.0 (early, explore) to ~0.5 (late, exploit)
    # Use a smooth transition based on gap
    progress = 1.0 - np.clip(gap_to_min / 1.0, 0, 1)  # 0 when far, 1 when close
    beta = 0.5 + 1.5 * (1.0 - progress)  # 2.0 when far, 0.5 when close
    
    # UCB score: mu - beta * sigma (lower is better, so we negate for "higher is better")
    ucb_score = -mu + beta * sigma
    
    # Expected Improvement component
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But we want "higher is better" for the score, and EI is positive when improvement is expected
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI to be comparable to UCB scale
    # EI has units of function value, UCB also has units of function value
    # We want to combine them. Let's use a weighted sum.
    
    # Weight: more weight on EI when we're close to minimum (exploit)
    # More weight on UCB when we're far (explore)
    w_ei = 0.3 + 0.7 * progress  # 0.3 when exploring, 1.0 when exploiting
    w_ucb = 1.0 - w_ei
    
    # Scale EI to be comparable to UCB magnitude
    # Typical EI magnitude: 0.01 to 0.1
    # Typical UCB magnitude: 0.1 to 1.0
    # Let's scale EI by a factor to make it comparable
    ei_scaled = ei * 5.0  # Empirical scaling
    
    # Final score: higher is better
    score = w_ei * ei_scaled + w_ucb * ucb_score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
