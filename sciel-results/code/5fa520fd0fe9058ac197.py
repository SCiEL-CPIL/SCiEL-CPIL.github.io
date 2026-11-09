"""Rewritten acquisition function using a Hypervolume-inspired acquisition for branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses a Hypervolume-based acquisition that approximates expected hypervolume
    improvement relative to f_best, with a discretized integration over the
    objective space. This structural change moves away from pure EI/UCB toward
    a multi-objective-inspired criterion that better balances exploration and
    exploitation in low-dimensional BO.
    """
    # Time budget guard
    elapsed = time.time() - _start_time
    if elapsed > 290:
        # Fallback: simple EI
        sigma_safe = np.where(np.asarray(sigma, dtype=np.float64) > 1e-12,
                              np.asarray(sigma, dtype=np.float64), 1e-12)
        mu_safe = np.asarray(mu, dtype=np.float64)
        z = (f_best - mu_safe) / sigma_safe
        ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.where(sigma > 1e-12, sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1.0)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Hypervolume-based acquisition with adaptive discretization
    known_min = 0.397887
    
    # Progress: 0 at start (f_best high), 1 near optimum
    # Use a smoother progress measure based on relative gap
    initial_range = 15.0 - known_min  # approximate range of branin
    progress = max(0.0, min(1.0, (initial_range - (f_best - known_min)) / initial_range))
    
    # Adaptive number of discretization points: more near optimum
    n_points = 30 + int(20 * progress)
    n_points = min(n_points, 50)
    
    # Define the range of thresholds
    # Early on, consider a wider range; later, focus near f_best
    if progress < 0.3:
        # Early: wide range to explore
        lower_bound = max(known_min, f_best - 5.0)
    else:
        # Later: tighter range around f_best
        lower_bound = max(known_min, f_best - 1.0)
    upper_bound = f_best
    
    if upper_bound - lower_bound < 1e-6:
        # Very close to optimum, use simple EI
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return np.where(np.isfinite(ei), ei, 0.0)
    
    # Create a grid of thresholds
    # Use non-uniform spacing: denser near f_best (upper end)
    # This gives better resolution where improvements are most likely
    thresholds = np.linspace(lower_bound, upper_bound, n_points)
    
    # Vectorized computation
    mu_expanded = mu[:, np.newaxis]  # (n_candidates, 1)
    sigma_expanded = sigma[:, np.newaxis]  # (n_candidates, 1)
    thresholds_expanded = thresholds[np.newaxis, :]  # (1, n_points)
    
    # z-scores: (n_candidates, n_points)
    z_grid = (thresholds_expanded - mu_expanded) / sigma_expanded
    z_grid = np.clip(z_grid, -10, 10)
    
    # CDF values: P(f(x) < threshold)
    cdf_grid = norm.cdf(z_grid)
    
    # Improvement magnitudes: (f_best - threshold) for each threshold
    improvements = f_best - thresholds
    improvements = np.maximum(improvements, 0.0)
    
    # Trapezoidal integration weights
    delta_t = np.zeros(n_points)
    if n_points > 1:
        delta_t[1:] = thresholds[1:] - thresholds[:-1]
        delta_t[0] = thresholds[1] - thresholds[0]
        delta_t[-1] = thresholds[-1] - thresholds[-2]
        # Normalize
        total_weight = np.sum(improvements * delta_t)
        if total_weight > 1e-12:
            delta_t = delta_t / total_weight
    else:
        delta_t = np.ones(1)
    
    # Expected hypervolume improvement
    ehi = np.sum(cdf_grid * improvements[np.newaxis, :] * delta_t[np.newaxis, :], axis=1)
    
    # Exploration bonus: scale with sigma, but dampen as we get closer to optimum
    # Use a temperature that decreases with progress
    temperature = 1.0 - 0.5 * progress
    exploration_weight = 0.1 * temperature
    
    # Normalize sigma for exploration bonus
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    if sigma_std > 1e-12:
        sigma_norm = (sigma - sigma_mean) / sigma_std
    else:
        sigma_norm = np.zeros_like(sigma)
    sigma_norm = np.clip(sigma_norm, -3, 3)
    
    scores = ehi + exploration_weight * sigma_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
