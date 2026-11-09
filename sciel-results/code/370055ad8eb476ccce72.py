"""Cost-aware acquisition function for Bayesian optimization on rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Set a deadline guard: we want to finish well within 330s
_DEADLINE = 310.0  # seconds
_start_time = time.time()


def _check_deadline():
    """Check if we're close to the deadline."""
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-6)
    
    # Clamp sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural change: Cost-aware Quantile of the Posterior (QoP)
    # QoP selects the p-quantile of the posterior predictive distribution,
    # which provides a principled balance between exploitation and exploration.
    # For minimization, we want the lower quantile (optimistic prediction).
    #
    # The p-quantile of N(mu, sigma^2) is: mu - sigma * z_p
    # where z_p = norm.ppf(1-p) for the lower p-quantile.
    #
    # We use a fixed quantile p=0.25 (i.e., 25th percentile), which was found
    # to work well in prior experiments. This gives a moderately optimistic
    # prediction that balances exploration and exploitation.
    #
    # To make it cost-aware, we divide by a power of the cost. Using exponent
    # 1.0 (linear cost penalty) gives a strong preference for cheaper evaluations,
    # which is important given the budget constraint of 30.
    
    # Compute the 25th percentile of the posterior (optimistic for minimization)
    # z_{0.25} = norm.ppf(0.25) ≈ -0.6745
    # So the 25th percentile = mu + sigma * norm.ppf(0.25) = mu - sigma * 0.6745
    z_p = norm.ppf(0.25)  # ≈ -0.6745
    qop = mu + sigma_safe * z_p  # Lower quantile (optimistic for minimization)
    
    # Cost-aware score: divide by cost to prefer cheaper evaluations
    # Use exponent 1.0 for linear cost penalty
    cost_exponent = 1.0
    cost_penalty = cost_safe ** cost_exponent
    
    scores = -qop / cost_penalty  # Negate because we want higher = better (lower qop is better)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
