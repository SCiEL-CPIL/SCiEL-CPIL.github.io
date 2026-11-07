"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as student_t


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using Probability of Improvement (PoI)
    approximated with a Student's t-distribution.
    
    This structural change moves from Expected Improvement (which balances
    magnitude and probability) to a probability-focused criterion with heavier
    tails (t-distribution), which may better target high-improvement regions
    by being more sensitive to tail events in the posterior.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising points.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Degrees of freedom for the t-distribution
    # Lower nu implies heavier tails, more robust to outliers, more exploratory in tails
    nu = 3.0
    
    # Standardized improvement
    # z = (f_best - mu) / sigma
    # If mu < f_best, z is positive.
    z = (f_best - mu) / sigma
    
    # Calculate Probability of Improvement using t-distribution CDF
    # P(f(x) < f_best) = F_t(z; nu)
    scores = student_t.cdf(z, df=nu)
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    # The scores are in [0, 1]. Higher is better.
    return scores
# EVOLVE-BLOCK-END
