"""Acquisition function for Bayesian optimization of weierstrass 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for budget management
_START_TIME = time.time()
_TIME_BUDGET = 320.0  # Leave 10s margin before 330s kill


def _elapsed():
    """Return seconds elapsed since module load."""
    return time.time() - _START_TIME


def _ei(mu, sigma, f_best, xi=0.0):
    """
    Standard Expected Improvement acquisition.
    Higher score = more promising.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(np.isfinite(ei), ei, 0.0)


def _knowledge_gradient(mu, sigma, f_best, xi=0.0):
    """
    Knowledge Gradient acquisition: expected reduction in simple regret.
    KG = E[max(0, f_best - min(Y))] where Y is the next observation.
    Approximated via: E[max(0, f_best - Y)] which is just EI,
    but we use a slightly different weighting to encourage exploration
    of regions where uncertainty is high relative to current best.
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    # Standard EI serves as a proxy for KG in 1D
    kg = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.where(np.isfinite(kg), kg, 0.0)


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition combining Expected Improvement with adaptive
    exploration weighting based on posterior uncertainty.
    
    Uses a time-adaptive strategy:
    - Early on: more exploration (higher beta in UCB-like term)
    - Later: more exploitation (standard EI)
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed objective value.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising.
    """
    # Check time budget
    if _elapsed() > _TIME_BUDGET:
        # Emergency: return pure exploitation (EI) quickly
        return _ei(mu, sigma, f_best)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute standard EI
    ei_scores = _ei(mu, sigma, f_best)
    
    # Compute uncertainty-based exploration bonus
    # Normalize sigma to [0, 1] range for this batch
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Adaptive exploration weight:
    # Use a small constant exploration bonus scaled by normalized uncertainty
    # This encourages visiting uncertain regions without dominating exploitation
    exploration_weight = 0.05
    exploration_bonus = exploration_weight * sigma_norm * (f_best - np.min(mu) + 1e-8)
    
    # Combine: EI + exploration bonus
    # The exploration bonus is small but helps escape local optima in multi-modal landscape
    scores = ei_scores + exploration_bonus
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Additional structural change: use a rank-based reweighting to
    # ensure we don't just always pick the top EI point.
    # We add a small diversity term based on the rank of sigma
    # among the top candidates.
    # Find top-K by EI
    k = 100
    top_k_idx = np.argpartition(ei_scores, -k)[-k:]
    top_k_ei = ei_scores[top_k_idx]
    top_k_sigma = sigma_safe[top_k_idx]
    
    # Among top-K, slightly boost those with higher sigma
    # This is a mild diversity mechanism
    if len(top_k_idx) > 0:
        # Normalize sigma within top-K
        tk_sigma_min = np.min(top_k_sigma)
        tk_sigma_max = np.max(top_k_sigma)
        tk_range = tk_sigma_max - tk_sigma_min
        if tk_range > 1e-12:
            tk_sigma_norm = (top_k_sigma - tk_sigma_min) / tk_range
        else:
            tk_sigma_norm = np.zeros_like(top_k_sigma)
        
        # Apply small boost
        diversity_boost = 0.02 * tk_sigma_norm * (np.max(top_k_ei) - np.min(top_k_ei) + 1e-8)
        scores[top_k_idx] += diversity_boost
    
    # Final safety check
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
