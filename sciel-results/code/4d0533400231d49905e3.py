import numpy as np
from scipy.stats import norm

_global_eval_count = 0

def _normalize_scores(scores):
    """Min-max normalize scores to [0, 1]."""
    min_s = np.min(scores)
    max_s = np.max(scores)
    if max_s - min_s < 1e-12:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)

def acquisition(mu, sigma, f_best):
    global _global_eval_count
    _global_eval_count += 1
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement (minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB (minimization): We want to minimize, so we prefer low mu and high sigma.
    # Standard UCB for minimization: mu - beta * sigma
    # As evals increase, we want to reduce exploration (lower beta).
    # Start with beta=3, decay to beta=1.
    # 30 total evals.
    beta = 3.0 * (1.0 - (_global_eval_count - 1) / 29.0) + 1.0
    ucb = mu - beta * sigma_safe
    
    # Normalize EI and UCB to [0, 1] for the current batch
    # Note: For minimization, higher EI is better. Higher UCB score (mu - beta*sigma) is "worse" in terms of value, 
    # but in acquisition context, we usually maximize the acquisition score.
    # Wait. 
    # EI: High value = Good.
    # UCB for minimization: We want to find low values. 
    # If we use `mu - beta * sigma`, points with low mu and high sigma will have low scores.
    # But acquisition functions are MAXIMIZED.
    # So for minimization, the UCB acquisition score should be something like `- (mu - beta * sigma)`? 
    # Or rather, we want to maximize the "confidence that this point is good".
    # A point is "good" if it has low mu.
    # Uncertainty helps if it might be even lower.
    # So we want to maximize `-(mu - beta * sigma)` = `-mu + beta * sigma`.
    # Let's call this `ucb_score`.
    
    ucb_score = -mu + beta * sigma_safe
    
    # Normalize
    ei_norm = _normalize_scores(ei)
    ucb_norm = _normalize_scores(ucb_score)
    
    # Blend
    # Early on, rely more on UCB (exploration). Later, rely on EI (exploitation).
    # We can blend the normalized scores.
    # Weight for EI increases with time.
    w_ei = 0.2 + 0.6 * ((_global_eval_count - 1) / 29.0)
    w_ucb = 1.0 - w_ei
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
