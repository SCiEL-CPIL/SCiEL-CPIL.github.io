"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a UCB exploration term, normalized by evaluation cost.
    
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
    # Adaptive beta: more exploration early, less as we converge
    f_best_abs = abs(f_best)
    beta = 0.5 + 1.5 * np.exp(-f_best_abs / 5.0)
    ucb = beta * sigma_safe - mu
    
    # Adaptive exploration weight: shift from exploration to exploitation
    # as f_best approaches the known minimum (0 for levy 2d)
    # When f_best is large, we need more exploration; when small, exploit
    w_ucb = 0.1 + 0.3 * np.exp(-f_best_abs / 2.0)
    
    # Soft-max: smooth approximation of max(ucb, 0) for better gradient flow
    tau = 0.1
    soft_ucb = np.where(ucb > 0,
                        ucb + tau * np.log1p(np.exp(-ucb / tau)),
                        tau * np.log1p(np.exp(ucb / tau)))
    
    # Combine: primary EI signal + secondary UCB exploration signal
    combined = ei + w_ucb * soft_ucb
    
    # Cost-aware: divide by cost
    score = combined / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
