"""Acquisition for Bayesian optimization of Hartmann 6d using adaptive UCB."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: evaluator kills at 330s wall-clock; leave a safety margin.
_DEADLINE = time.time() + 325.0

# Known minimum of the Hartmann 6d function.
_HARTMANN_6D_MIN = -3.32237

# Hartmann 6d range is roughly [-3.32, 0.6], so span ~4.
_HARTMANN_6D_SPAN = 4.0


def _adaptive_ucb(mu, sigma, f_best):
    """
    Adaptive UCB for minimization with smooth exploration-exploitation transition.
    
    beta(t) = beta_min + (beta_max - beta_min) * exp(-decay * gap)
    where gap = f_best - f_min (distance to known optimum).
    
    Early on (large gap): beta is high -> explore uncertain regions.
    Late (small gap): beta is low -> exploit promising regions.
    
    The exponential decay gives a smooth, non-linear transition that
    allows sustained exploration when far from the optimum and rapid
    exploitation as we converge.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Gap from known optimum
    gap = max(f_best - _HARTMANN_6D_MIN, 0.0)
    
    # Normalize gap to [0, 1]
    gap_norm = min(gap / _HARTMANN_6D_SPAN, 1.0)
    
    # Adaptive beta: exponential decay from high exploration to low exploitation
    # When gap_norm = 1 (far from optimum): beta = beta_max
    # When gap_norm = 0 (at optimum): beta = beta_min
    beta_min = 0.3
    beta_max = 3.5
    decay = 2.5  # controls how quickly exploration fades as we approach optimum
    beta = beta_min + (beta_max - beta_min) * np.exp(-decay * (1.0 - gap_norm))
    
    # UCB for minimization: score = (f_best - mu) + beta * sigma
    # High when mu is low (good prediction) and sigma is high (uncertain)
    scores = (f_best - mu) + beta * sigma_safe
    
    return scores


def _ei_with_adaptive_exploration(mu, sigma, f_best):
    """
    Expected Improvement with adaptive exploration weight.
    
    EI naturally balances exploration and exploitation, but in high dimensions
    it can be too conservative early on. We add a sigma term whose weight
    decreases as we get closer to the known optimum.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Gap from known optimum
    gap = max(f_best - _HARTMANN_6D_MIN, 0.0)
    gap_norm = min(gap / _HARTMANN_6D_SPAN, 1.0)
    
    # Exploration weight: high when far from optimum, low when close
    # Smooth transition using sigmoid-like function
    w_explore = 0.15 + 0.85 * gap_norm  # ranges from 0.15 to 1.0
    
    scores = ei + w_explore * sigma_safe
    return scores


def _knowledge_gradient_adaptive(mu, sigma, f_best):
    """
    Adaptive Knowledge Gradient: combines EI with a component that
    rewards reducing uncertainty about the global minimum.
    
    The idea: when f_best is far from the known optimum, we should
    prioritize points that are likely to significantly improve f_best.
    When close, we should be more conservative.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Gap from known optimum
    gap = max(f_best - _HARTMANN_6D_MIN, 0.0)
    gap_norm = min(gap / _HARTMANN_6D_SPAN, 1.0)
    
    # KG-like: weight EI by how much improvement is still possible
    # When gap is large, each unit of improvement is more valuable
    kg_weight = 1.0 + 0.5 * gap_norm
    
    # Additional exploration for high-uncertainty regions
    # Scaled by gap: more exploration when far from optimum
    w_explore = 0.05 + 0.3 * gap_norm
    
    scores = kg_weight * ei + w_explore * sigma_safe
    return scores


def _blend_acquisitions(mu, sigma, f_best):
    """
    Blend multiple acquisition functions with adaptive weights.
    
    Uses a soft mixture: each acquisition is weighted by how well-suited
    it is to the current stage of optimization (early vs late).
    """
    n = mu.shape[0]
    if n == 0:
        return np.zeros(0)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Gap from known optimum
    gap = max(f_best - _HARTMANN_6D_MIN, 0.0)
    gap_norm = min(gap / _HARTMANN_6D_SPAN, 1.0)
    
    # Compute each acquisition
    ucb_scores = _adaptive_ucb(mu, sigma, f_best)
    ei_scores = _ei_with_adaptive_exploration(mu, sigma, f_best)
    kg_scores = _knowledge_gradient_adaptive(mu, sigma, f_best)
    
    # Adaptive weights: early on, favor UCB (exploration); late, favor EI/KG (exploitation)
    # Use smooth interpolation
    w_ucb = 0.3 + 0.5 * gap_norm       # 0.3 to 0.8
    w_ei = 0.4 + 0.3 * (1.0 - gap_norm) # 0.4 to 0.7
    w_kg = 0.3 + 0.2 * (1.0 - gap_norm) # 0.3 to 0.5
    
    # Normalize weights
    w_sum = w_ucb + w_ei + w_kg
    w_ucb /= w_sum
    w_ei /= w_sum
    w_kg /= w_sum
    
    # Blend
    scores = w_ucb * ucb_scores + w_ei * ei_scores + w_kg * kg_scores
    
    return scores


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6d.
    
    Uses an adaptive blend of UCB, EI, and KG with weights that shift
    from exploration-heavy (early) to exploitation-heavy (late) based
    on the current gap to the known global minimum.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Deadline check: if we're too close to the limit, return a fast fallback
    if time.time() > _DEADLINE - 2.0:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei + 0.2 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    try:
        scores = _blend_acquisitions(mu, sigma, f_best)
    except Exception:
        # Fallback on any error: simple EI + moderate exploration
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei + 0.2 * sigma_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)

    # Final sanity: ensure shape matches and all finite
    if scores.shape != mu.shape:
        scores = np.resize(scores, mu.shape)
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
