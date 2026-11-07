"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 325.0  # Safety margin vs 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of 1D sphere function.
    
    Combines:
    1. Expected Improvement (EI) for exploitation
    2. Posterior quantile (Q_{1-alpha}) for stochastic lower-bound targeting
    3. UCB for exploration
    
    Dynamically blends based on uncertainty and proximity to optimum.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)
    
    # Deadline check
    if time.time() > _DEADLINE - 1.0:
        return np.zeros(len(mu))
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-10)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-10)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 1e10)
    
    # 1. Expected Improvement (EI) for minimization
    z = (fb - mu) / sigma
    z_clipped = np.clip(z, -50, 50)
    
    ei = (fb - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)
    
    # 2. Posterior quantile Q_{1-alpha} for stochastic lower-bound targeting
    # With alpha = 1e-4, z_alpha = norm.ppf(1 - alpha) ≈ 3.719
    alpha = 1e-4
    z_alpha = norm.ppf(1.0 - alpha)  # ≈ 3.719
    
    # The quantile represents the expected minimum of many samples from the posterior
    # For minimization, we want to maximize the negative of this quantile
    # Q_{1-alpha}(x) = mu(x) + z_alpha * sigma(x)
    # Score to maximize: -Q_{1-alpha}(x) = -mu(x) - z_alpha * sigma(x)
    # But this is overly conservative. Instead, use it as an exploration term.
    # We want candidates where the lower quantile is low (good potential minimum)
    quantile_score = -mu - z_alpha * sigma
    
    # 3. UCB for exploration
    beta = 2.0
    ucb_scores = beta * sigma - mu
    
    # Normalize all scores to [0, 1] range for blending
    def normalize_scores(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_max > arr_min:
            return (arr - arr_min) / (arr_max - arr_min)
        else:
            return np.zeros_like(arr)
    
    ei_norm = normalize_scores(ei)
    quantile_norm = normalize_scores(quantile_score)
    ucb_norm = normalize_scores(ucb_scores)
    
    # Dynamic weighting based on uncertainty and proximity to optimum
    mean_sigma = np.mean(sigma)
    
    # Sigmoid-like transition based on mean_sigma
    x = mean_sigma / 1.0
    w_ucb = 1.0 / (1.0 + np.exp(-x))
    
    # Dampen exploration if f_best is very small (close to optimum)
    if fb < 0.1:
        w_ucb *= 0.3
    elif fb < 0.5:
        w_ucb *= 0.6
    
    # Weight for quantile-based exploration (targets lower bound)
    # Use a fixed weight that decays as we get closer to optimum
    w_quantile = 0.2 * max(0.0, 1.0 - fb / 10.0)
    
    # Weight for EI (exploitation)
    w_ei = 1.0 - w_ucb - w_quantile
    w_ei = np.clip(w_ei, 0.0, 1.0)
    w_ucb = np.clip(w_ucb, 0.0, 1.0)
    w_quantile = np.clip(w_quantile, 0.0, 1.0)
    
    # Ensure weights sum to 1
    weight_sum = w_ei + w_ucb + w_quantile
    if weight_sum > 0:
        w_ei /= weight_sum
        w_ucb /= weight_sum
        w_quantile /= weight_sum
    
    # Blend scores
    scores = w_ei * ei_norm + w_ucb * ucb_norm + w_quantile * quantile_norm
    
    # Rescale to a reasonable magnitude
    max_original = max(np.max(ei), np.max(ucb_scores), np.max(quantile_score))
    if max_original > 0:
        scores = scores * max_original
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
