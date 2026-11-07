"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining expected improvement
    with a log-barrier cost penalty and sublinear exploration boost.
    
    Parameters
    ----------
    mu : array_like, shape (2000,)
        GP mean predictions in objective units.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Lowest observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean of scaled coords).
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = prefer to evaluate).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip EI to non-negative (EI can be slightly negative due to numerical issues)
    ei = np.maximum(ei, 0.0)
    
    # Log-barrier form: PI * sigma^gamma / cost^delta
    # gamma < 1 provides sublinear exploration boost (more balanced across uncertainty scales)
    # delta < 1 softens the cost penalty (allows visiting slightly more expensive but informative regions)
    gamma = 0.7
    delta = 0.5
    
    # Exploration term: sigma^gamma
    exploration = np.power(sigma_safe, gamma)
    
    # Cost penalty: 1 / cost^delta
    cost_penalty = np.power(np.maximum(cost, 1e-9), -delta)
    
    # Combined score: EI * exploration * cost_penalty
    # This structurally changes the exploitation-exploration tradeoff:
    # - EI captures exploitation (expected improvement)
    # - sigma^gamma provides sublinear exploration boost
    # - cost^(-delta) softly penalizes expensive evaluations
    scores = ei * exploration * cost_penalty
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
