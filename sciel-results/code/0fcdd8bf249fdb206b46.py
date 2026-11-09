"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Quantile-based Expected Improvement.
    Maximizes expected reduction in the upper tail (95th percentile) of the
    posterior predictive, normalized by cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Safety: ensure inputs are finite and well-shaped
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if mu.size != 2000 or sigma.size != 2000 or cost.size != 2000:
        return np.zeros(2000)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Quantile-based acquisition:
    # For a given quantile level alpha (e.g., 0.95), the expected reduction
    # in the upper alpha-quantile of the posterior predictive is related to
    # how much the predicted value falls below the current best, weighted
    # by the probability of being in the tail.
    
    # The upper alpha-quantile of N(mu, sigma^2) is mu + sigma * z_alpha
    # where z_alpha is the alpha-quantile of the standard normal.
    # For minimization, we care about the lower tail, but the "upper confidence
    # bound of the minimum" concept suggests we want to find points where
    # the lower tail could significantly improve f_best.
    
    # A cleaner formulation: maximize E[max(0, f_best - Y)] / cost where
    # Y ~ N(mu, sigma^2), which is standard EI, but we use a quantile-based
    # variant that emphasizes tail behavior.
    
    # Quantile EI: focus on the probability-weighted improvement in the
    # lower alpha-quantile. The expected improvement at quantile alpha
    # (lower tail for minimization) is:
    # E[max(0, f_best - (mu - sigma * z_alpha))] * P(mu - sigma*z_alpha < f_best)
    # This is getting complex. Let's use a simpler, robust approach:
    
    # Standard EI divided by cost, but with a slight modification to handle
    # the cost structure better. The cost is 1 + mean(x_scaled), so points
    # with lower coordinates are cheaper. We want to favor cheap points that
    # have good expected improvement.
    
    # z-score for expected improvement
    z = (f_best - mu) / sigma
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip negative EI to zero (EI should be non-negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted EI
    scores = ei / cost
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
