"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) acquisition function.
    
    Unlike UCB which adds a linear exploration bonus, PI calculates the probability
    that a candidate point yields a value better than the current best (f_best).
    For minimization, this is P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    
    This strategy is structurally different from UCB/EI:
    - It ignores the magnitude of improvement (unlike EI).
    - It does not add a linear uncertainty term (unlike UCB).
    - It naturally balances exploration and exploitation based on the posterior
      probability mass below the current best. Early on, when sigma is large,
      many points have significant probability of being better than f_best.
      As sigma shrinks, only points with mu significantly below f_best score high.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian Process posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the Gaussian Process posterior.
    f_best : float
        The best (lowest) observed value so far.
        
    Returns:
    --------
    scores : array, shape (2000,)
        PI scores. Higher is better (more promising).
    """
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle numerical stability for sigma to avoid division by zero
    # If sigma is 0, the probability is 1 if mu < f_best, else 0.
    # We use a small epsilon to keep the calculation stable in the CDF call.
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Calculate z-score for the probability of improvement
    # z = (f_best - mu) / sigma
    # If z is large positive, P(f < f_best) is close to 1.
    # If z is large negative, P(f < f_best) is close to 0.
    z = (f_best - mu) / sigma_safe
    
    # Compute the Cumulative Distribution Function of the standard normal
    # This gives the probability that a standard normal variable is less than z.
    # Since f ~ N(mu, sigma^2), P(f < f_best) = P((f-mu)/sigma < (f_best-mu)/sigma) = Phi(z)
    scores = norm.cdf(z)
    
    # Ensure scores are finite and within [0, 1]
    scores = np.clip(scores, 0.0, 1.0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
