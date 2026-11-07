import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 320.0  # 320s to leave safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a hybrid of Expected Improvement (EI) and
    Upper Confidence Bound (UCB) for minimization of Weierstrass 1D.
    
    Parameters:
    mu (np.ndarray): Posterior mean at candidate points, shape (2000,)
    sigma (np.ndarray): Posterior standard deviation at candidate points, shape (2000,)
    f_best (float): Best observed function value so far
    
    Returns:
    np.ndarray: Scores for each candidate point, shape (2000,). Higher score = evaluate next.
    """
    # Deadline guard: if we're too close to the limit, use a fast deterministic fallback
    if time.time() > _DEADLINE - 5.0:
        sigma_safe = np.maximum(sigma, 1e-9)
        return -(mu - 0.1 * sigma_safe)
    
    # Ensure sigma is non-negative and numerically stable
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in extreme tails
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB score for minimization
    # For minimization, UCB explores regions with high uncertainty
    # Adaptive kappa: start high for exploration, decrease as f_best improves
    # f_best for Weierstrass on [-0.5, 0.5] is in [0, ~1]
    f_best_norm = max(0.0, min(1.0, f_best / 1.0))
    
    # Start with high kappa (~4.0) for exploration, decrease to ~1.0 as we approach optimum
    kappa = 1.0 + 3.0 * (1.0 - f_best_norm)
    
    # UCB for minimization: we want to minimize, so we maximize -(mu + kappa * sigma)
    # But we also want to exploit low mu, so use: score = -(mu + kappa * sigma)
    ucb_score = -(mu + kappa * sigma)
    
    # Normalize both scores to have zero mean and unit variance for robust combination
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        z_ei = np.zeros_like(ei)
    else:
        z_ei = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score)
    if ucb_std < 1e-9:
        z_ucb = np.zeros_like(ucb_score)
    else:
        z_ucb = (ucb_score - ucb_mean) / ucb_std
    
    # Adaptive weighting: more exploration early (high mean sigma), more exploitation late
    # Use mean sigma as proxy for exploration level
    mean_sigma = np.mean(sigma)
    max_sigma = np.max(sigma)
    sigma_norm = np.clip(mean_sigma / (max_sigma + 1e-10), 0.0, 1.0)
    
    # When uncertainty is high (early), weight UCB more; when low (late), weight EI more
    w_explore = 0.3 + 0.5 * sigma_norm  # ranges from 0.3 to 0.8
    w_exploit = 1.0 - w_explore
    
    # Combine normalized scores
    scores = w_exploit * z_ei + w_explore * z_ucb
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
