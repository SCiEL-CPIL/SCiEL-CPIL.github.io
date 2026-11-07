"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard (330 s wall-clock limit; keep a safety margin)
_T0 = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0  # seconds before the hard limit


def _within_deadline():
    return (time.time() - _T0) < (_DEADLINE - _MARGIN)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on Probability of Improvement (PI) 
    with a dynamic exploration bonus.
    
    This structural change moves away from Expected Improvement (EI) 
    which can get stuck in local minima, towards PI which focuses on 
    the probability of finding a new best, encouraging broader exploration 
    in multimodal landscapes like Styblinski-Tang.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]

    # Numerical safeguards
    eps = 1e-12
    sigma = np.maximum(sigma, eps)

    # 1. Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma
    # Clip z to prevent numerical issues in CDF
    z = np.clip(z, -10.0, 10.0)
    pi_scores = norm.cdf(z)
    
    # 2. Exploration Term
    # We want to explore high uncertainty regions. 
    # Instead of a linear UCB, we use a normalized sigma score.
    # We normalize sigma by its max to keep it in [0, 1].
    sigma_max = np.max(sigma)
    if sigma_max < eps:
        sigma_norm = np.zeros_like(sigma)
    else:
        sigma_norm = sigma / sigma_max
        
    # 3. Dynamic Exploration Weight
    # The weight for exploration should be high when we are early on 
    # (high global uncertainty) and low when we are late (low global uncertainty).
    # We use the mean sigma as a proxy for global uncertainty.
    mean_sigma = np.mean(sigma)
    
    # Map mean_sigma to a weight in [0, 1].
    # If mean_sigma is large, we explore more.
    # We use a sigmoid-like transition centered around a threshold.
    # The threshold is set relative to the initial scale of the problem.
    # For ST-1D, the range is roughly [-40, 0]. Initial sigma might be ~1-5.
    # Let's use a threshold of 0.5 for the normalized sigma or absolute sigma?
    # Let's use a softplus-like decay.
    # If mean_sigma is > 1.0, weight is high. If < 0.1, weight is low.
    # weight = 1 / (1 + exp(-k * (mean_sigma - threshold)))
    k = 5.0
    threshold = 0.5
    explore_weight = 1.0 / (1.0 + np.exp(-k * (mean_sigma - threshold)))
    
    # Cap the exploration weight to not overwhelm PI completely
    explore_weight = np.clip(explore_weight, 0.0, 0.8)
    
    # 4. Combine
    # Scores = (1 - explore_weight) * PI + explore_weight * Sigma_Norm
    # This ensures that even if PI is low, high sigma points get a chance.
    # And if PI is high, it dominates.
    
    scores = (1.0 - explore_weight) * pi_scores + explore_weight * sigma_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Add a tiny random noise to break ties? 
    # The task says "returns 2000 finite scores". 
    # Adding noise can help if there are exact ties, but GP predictions are continuous.
    # We'll stick to deterministic for stability unless needed.
    
    return scores
# EVOLVE-BLOCK-END
