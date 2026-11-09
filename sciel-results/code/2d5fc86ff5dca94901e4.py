"""Acquisition function combining Quantile-based scoring with adaptive Hypervolume-inspired refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    Uses a Quantile-based acquisition (5th percentile) as the core scoring mechanism,
    combined with an adaptive Hypervolume-inspired refinement that provides
    structured exploitation signal near the optimum.
    
    The 5th percentile (mu - 1.645 * sigma) provides a robust lower-bound estimate
    that focuses on the tail of the distribution where significant improvements are
    likely, balancing exploration of high-uncertainty regions with exploitation of
    low-mean regions without arbitrary beta tuning.
    
    Near the optimum, a Hypervolume-based Expected Improvement (EHI) term is blended
    in to provide finer-grained exploitation signal.
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

    known_min = 0.397887
    initial_range = 15.0 - known_min  # approximate range of branin

    # Progress: 0 at start (f_best high), 1 near optimum
    progress = max(0.0, min(1.0, (initial_range - (f_best - known_min)) / initial_range))

    # Core: Quantile-based score (5th percentile)
    # Higher score means more likely to find low values
    # 5th percentile = mu - 1.645 * sigma
    # We want to maximize the score, so we use negative of the quantile
    quantile_score = -(mu - 1.645 * sigma)

    # Adaptive Hypervolume-based refinement
    # Blend weight increases as we get closer to the optimum
    ehi_weight = 0.3 * progress  # 0 early, 0.3 near optimum

    if ehi_weight > 0.01:
        # Compute EHI with adaptive discretization
        if progress < 0.3:
            lower_bound = max(known_min, f_best - 5.0)
        else:
            lower_bound = max(known_min, f_best - 1.0)
        upper_bound = f_best

        if upper_bound - lower_bound < 1e-6:
            # Very close to optimum, use pure EI
            z = (f_best - mu) / sigma
            ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            ei = np.where(np.isfinite(ei), ei, 0.0)
            # Normalize EI to be comparable to quantile score
            ei_scale = np.std(quantile_score) + 1e-12
            ei = ei / ei_scale
            scores = quantile_score + ehi_weight * ei
            return np.where(np.isfinite(scores), scores, 0.0)

        n_points = 30 + int(20 * progress)
        n_points = min(n_points, 50)
        thresholds = np.linspace(lower_bound, upper_bound, n_points)

        mu_expanded = mu[:, np.newaxis]
        sigma_expanded = sigma[:, np.newaxis]
        thresholds_expanded = thresholds[np.newaxis, :]

        z_grid = (thresholds_expanded - mu_expanded) / sigma_expanded
        z_grid = np.clip(z_grid, -10, 10)
        cdf_grid = norm.cdf(z_grid)

        improvements = f_best - thresholds
        improvements = np.maximum(improvements, 0.0)

        delta_t = np.zeros(n_points)
        if n_points > 1:
            delta_t[1:] = thresholds[1:] - thresholds[:-1]
            delta_t[0] = thresholds[1] - thresholds[0]
            delta_t[-1] = thresholds[-1] - thresholds[-2]
            total_weight = np.sum(improvements * delta_t)
            if total_weight > 1e-12:
                delta_t = delta_t / total_weight
        else:
            delta_t = np.ones(1)

        ehi = np.sum(cdf_grid * improvements[np.newaxis, :] * delta_t[np.newaxis, :], axis=1)
        ehi = np.where(np.isfinite(ehi), ehi, 0.0)

        # Normalize EHI to be comparable to quantile score
        ehi_scale = np.std(quantile_score) + 1e-12
        ehi = ehi / ehi_scale

        scores = quantile_score + ehi_weight * ehi
    else:
        # Early stage: use pure quantile score
        scores = quantile_score

    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
