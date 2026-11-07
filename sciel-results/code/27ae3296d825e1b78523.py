"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as t_dist
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    
    Uses Student's t-distribution based Expected Improvement with a small
    exploration bonus. The t-distribution (nu=5) accounts for heavy-tailed
    uncertainty inherent in the limited 30-sample budget, preventing
    over-confidence in the GP mean and promoting more robust exploration.
    """
    t0 = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        return np.zeros(len(mu), dtype=np.float64)
    
    sigma = np.maximum(sigma, 1e-8)
    
    # Degrees of freedom for Student's t
    nu = 5.0
    
    # Standardized variable for t-distribution
    # Under t, if X ~ t_nu, then z = (f_best - mu) / (sigma * sqrt(nu/(nu-2)))
    # is the standardized variable for the t-CDF
    # Actually, for t-distribution with mean mu and std sigma:
    # P(X < x) = t_cdf((x - mu) / sigma, df=nu)
    # So for improvement: we want P(X < f_best) = t_cdf((f_best - mu) / sigma, df=nu)
    
    z = (f_best - mu) / sigma
    
    # t-distribution CDF and PDF
    t_cdf_val = t_dist.cdf(z, df=nu)
    t_pdf_val = t_dist.pdf(z, df=nu)
    
    # Expected Improvement for minimization with t-distribution:
    # EI = E[max(f_best - X, 0)] where X ~ N(mu, sigma^2) approximated by t
    # For t-distribution: EI = (f_best - mu) * t_cdf(z) + sigma * t_pdf(z) * (nu-1)/(nu-2)
    # The correction factor (nu-1)/(nu-2) accounts for the variance of t
    # Actually, the exact EI for t is more complex. Let's use:
    # EI = (f_best - mu) * t_cdf(z) + sigma * t_pdf(z)
    # This is an approximation that works well in practice
    
    ei = (f_best - mu) * t_cdf_val + sigma * t_pdf_val
    
    # The t-distribution has heavier tails, so it naturally provides more
    # exploration. We can use a slightly smaller exploration bonus.
    exploration_term = 0.05 * sigma
    
    scores = ei + exploration_term
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    elapsed = time.time() - t0
    if elapsed > 280.0:
        pass
    
    return scores
# EVOLVE-BLOCK-END
