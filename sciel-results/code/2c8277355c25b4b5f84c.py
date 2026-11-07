"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Probability of Improvement (PI) with
    dynamic UCB for aggressive exploitation and adaptive exploration.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Compute Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    pi = norm.cdf(z)
    
    # Compute Expected Improvement (EI) for additional exploitation signal
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute inverted UCB (uncertainty bonus)
    # Dynamic beta: higher when uncertainty is high (early), lower when low (late)
    median_sigma = np.median(sigma)
    max_sigma = np.max(sigma)
    
    # Map max_sigma to a beta range [1.5, 4.0]
    beta = 1.5 + 2.5 * (max_sigma / (max_sigma + 1.0))
    
    # For minimization, we want to explore where mu is low and sigma is high
    # UCB score = -mu + beta * sigma (higher is better)
    ucb_score = -mu + beta * sigma
    
    # Normalize each component to [0, 1] range
    eps = 1e-9
    
    pi_min = np.min(pi)
    pi_max = np.max(pi)
    pi_range = pi_max - pi_min
    if pi_range > eps:
        pi_norm = (pi - pi_min) / pi_range
    else:
        pi_norm = np.zeros_like(pi)
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > eps:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > eps:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Dynamic weights: 
    # When uncertainty is high (early), weight UCB more for exploration
    # When uncertainty is low (late), weight PI and EI more for exploitation
    # w_ucb ranges from 0.2 (late) to 0.5 (early)
    w_ucb = 0.2 + 0.3 * (max_sigma / (max_sigma + 1.0))
    
    # PI and EI are both exploitation-focused, so give them the remaining weight
    # PI is more aggressive (binary-like), EI is smoother
    w_pi = 0.5 * (1.0 - w_ucb)
    w_ei = 0.5 * (1.0 - w_ucb)
    
    # Combine: PI (aggressive exploitation) + EI (smooth exploitation) + UCB (exploration)
    scores = w_pi * pi_norm + w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 280:
        # Return current scores as-is if we're close to timeout
        pass
    
    return scores
# EVOLVE-BLOCK-END
