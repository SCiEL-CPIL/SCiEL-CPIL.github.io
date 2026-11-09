"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Quantile Acquisition Function (QAF) for minimization.
    
    Returns scores where higher means more promising to evaluate next.
    We want to maximize the probability of finding a point with f < f_best - delta.
    Using the 95th percentile upper bound: score = (f_best - mu) - k * sigma
    where k is the 95th percentile of the standard normal (approx 1.645).
    
    This is equivalent to a Lower Confidence Bound (LCB) with beta = 1.645,
    which balances exploitation (low mu) and exploration (high sigma).
    
    To handle numerical issues:
    - Clip sigma to avoid division by zero or negative values
    - Ensure all scores are finite
    """
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # 95th percentile of standard normal distribution
    k = 1.6448536269514722  # norm.ppf(0.95)
    
    # QAF score: we want to find points where f < f_best
    # The expected improvement in the "quantile" sense is:
    # P(f < f_best) is high when mu is low and sigma is small
    # But we also want to explore uncertain regions
    
    # Using the LCB formulation: score = mu + k * sigma (for minimization, we minimize this)
    # But the task says "higher meaning evaluate next", so we negate:
    # score = -(mu + k * sigma)
    
    # Actually, let's think about this more carefully.
    # For minimization, we want to find points with low f.
    # The LCB for minimization is: mu - k * sigma (lower bound on the true value)
    # We want to maximize the probability of improvement, so we want to minimize mu - k*sigma
    # Since higher score means evaluate next, we return: -(mu - k*sigma) = -mu + k*sigma
    
    # Wait, let me reconsider. The standard EI for minimization is:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # This is always non-negative and higher means more expected improvement.
    
    # The QAF approach: maximize P(f < f_best - delta) for some delta
    # P(f < f_best - delta) = Phi((f_best - delta - mu) / sigma)
    # This is monotonic in (f_best - delta - mu) / sigma
    # So maximizing this is equivalent to maximizing (f_best - mu) / sigma
    
    # But this can be unstable. Let's use a hybrid approach:
    # Combine the probability of improvement with the expected improvement
    
    # Let's use the standard EI but with a small exploration bonus
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima
    exploration_bonus = 0.1 * sigma
    
    score = ei + exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
