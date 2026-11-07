"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: kill after 320s to stay safely under 330s
_DEADLINE = time.time() + 320.0


def _check_deadline():
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge-Gradient-inspired acquisition with sublinear cost discount.
    
    Uses a hybrid of Expected Improvement (exploitation) and optimistic
    improvement (exploration), weighted by local uncertainty, and scaled
    by 1/sqrt(cost) for cost-awareness.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - lowest observed value
        cost: array of shape (2000,) - evaluation costs
    
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: return a simple fallback if we're close to timeout
    if not _check_deadline():
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma
        scores = (f_best - mu) / np.sqrt(cost) + 0.5 * sigma / np.sqrt(cost)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (minimization)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Optimistic improvement: f_best - (mu - beta*sigma)
    # This represents the best-case improvement if f(x) could be as low as mu - beta*sigma
    beta = 1.0
    optimistic_improvement = f_best - (mu - beta * sigma_safe)
    optimistic_improvement = np.maximum(optimistic_improvement, 0.0)
    
    # Adaptive weighting based on local uncertainty
    # When sigma is high relative to the mean, favor exploration (optimistic)
    # When sigma is low, favor exploitation (EI)
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 1e-9:
        sigma_rel = sigma_safe / sigma_mean
    else:
        sigma_rel = np.zeros_like(sigma_safe)
    
    # Blend factor: more exploration when uncertain
    # Clip to [0, 1] and use a smooth transition
    blend = np.clip(0.3 + 0.4 * sigma_rel, 0.1, 0.9)
    
    # Combine EI and optimistic improvement
    score_benefit = (1 - blend) * ei + blend * optimistic_improvement
    
    # Sublinear cost penalty: 1/sqrt(cost) allows exploration of high-cost regions
    # while still preferring cheaper evaluations
    cost_penalty = 1.0 / np.sqrt(cost_safe)
    
    # Final score
    score = score_benefit * cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
