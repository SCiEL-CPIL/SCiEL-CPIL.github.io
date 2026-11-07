"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid EI + UCB acquisition function with cost awareness.
    
    Combines Expected Improvement (exploitation) with an Upper Confidence
    Bound component (exploration), normalized by evaluation cost.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Calculate z-score: (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # UCB component for exploration: rewards high uncertainty
    # beta controls the exploration-exploitation balance
    beta = 1.5
    ucb = beta * sigma_safe - mu
    
    # Weighted combination: EI (exploitation) + UCB (exploration)
    # The UCB term helps discover new regions when EI is low
    # We normalize EI by a scale factor to make it comparable to UCB
    mu_scale = max(float(np.max(np.abs(mu))) if mu.size > 0 else 1.0, 1e-3)
    ei_normalized = ei / max(mu_scale, 1e-6)
    
    # Combine: primary EI signal + secondary UCB exploration signal
    # EI is in objective units, UCB is also in objective units
    # We use a small weight for UCB to not overwhelm exploitation
    w_ucb = 0.3
    combined = ei + w_ucb * np.maximum(ucb, 0.0)
    
    # Cost-aware: divide by cost
    score = combined / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
