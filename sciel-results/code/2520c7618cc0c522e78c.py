# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) for minimization.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - best (lowest) observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Z-score for the improvement
    # We want to minimize f, so improvement is f_best - f(x)
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(-z) + sigma * phi(z)
    # Note: Phi(-z) is the CDF of the standard normal at -z
    # phi(z) is the PDF of the standard normal at z
    
    # Calculate terms
    # np.clip to avoid numerical issues in norm.cdf for very large/small z?
    # norm.cdf handles large z well, but let's be safe.
    cdf_neg_z = norm.cdf(-z)
    pdf_z = norm.pdf(z)
    
    scores = (f_best - mu) * cdf_neg_z + sigma_safe * pdf_z
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
