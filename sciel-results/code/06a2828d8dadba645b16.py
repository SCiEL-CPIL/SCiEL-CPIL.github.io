"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Scale-adaptive Expected Improvement with an IQR-based exploration parameter.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means evaluate next.
    """
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Guard against non-finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, np.nanmax(mu[np.isfinite(mu)]) if np.any(np.isfinite(mu)) else 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute IQR-based adaptive exploration parameter xi
    # This balances exploration and exploitation based on the current posterior uncertainty
    finite_mu = mu[np.isfinite(mu)]
    if len(finite_mu) > 0:
        q75, q25 = np.percentile(finite_mu, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-10:
            iqr = np.std(finite_mu) if np.std(finite_mu) > 1e-10 else 1e-6
    else:
        iqr = 1e-6
    
    # xi scales with the spread of the posterior mean, encouraging exploration
    # when uncertainty in the mean is high
    xi = 0.01 * iqr
    
    # Expected Improvement with exploration parameter xi
    # EI = E[max(f_best - f(x) - xi, 0)]
    # For f ~ N(mu, sigma^2):
    # EI = (f_best - mu - xi) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu - xi) / sigma
    
    z = (f_best - mu - xi) / sigma
    
    # Numerical stability: clip z to avoid overflow in cdf/pdf
    z = np.clip(z, -50, 50)
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    scores = (f_best - mu - xi) * cdf_z + sigma * pdf_z
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
