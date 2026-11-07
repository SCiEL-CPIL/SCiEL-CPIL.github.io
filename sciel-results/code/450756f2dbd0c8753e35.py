"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hypervolume-based acquisition function for Bayesian optimization of Branin 2D.
    
    This acquisition function approximates the expected hypervolume improvement
    relative to the current best value f_best. It balances exploration and
    exploitation by considering both the probability of improvement and the
    magnitude of potential improvement, with a non-linear weighting that
    favors larger improvements in regions where improvement is likely.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        The best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores, higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score: how many standard deviations below f_best the mean is
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # Expected Improvement (EI) component
    ei = (f_best - mu) * cdf + sigma * pdf
    
    # Hypervolume-inspired weighting: 
    # The idea is that hypervolume improvement depends not just on whether we improve
    # but on how much we improve relative to the current reference point.
    # We use a non-linear transformation that gives more weight to larger improvements.
    
    # Scale factor based on the spread of sigma to normalize the improvement
    sigma_mean = np.mean(sigma)
    if sigma_mean < 1e-9:
        sigma_mean = 1e-9
    
    # Normalize EI by typical sigma scale
    normalized_ei = ei / (sigma_mean + 1e-9)
    
    # Hypervolume-inspired score: combine EI with a term that rewards
    # regions where there's both high probability of improvement AND
    # large potential improvement. This is approximated by a product-like
    # structure that captures the joint contribution to hypervolume.
    
    # The expected hypervolume contribution can be approximated as:
    # E[max(f_best - f(x), 0)^alpha] for alpha > 1 to emphasize larger improvements
    # We approximate this using a moment-based approach.
    
    # For a Gaussian, E[max(f_best - f(x), 0)] = EI (first moment)
    # For higher moments, we can use:
    # E[max(f_best - f(x), 0)^2] = (f_best - mu)^2 * cdf + 2*sigma^2 * (z * cdf - pdf) + sigma^2 * (2*z*pdf + (z^2-1)*cdf) ... 
    # This gets complex. Instead, we use a simpler heuristic inspired by hypervolume.
    
    # Hypervolume-inspired score:
    # score = EI * (1 + beta * pdf / (cdf + eps))
    # This upweights points where the probability density is high relative to the CDF,
    # which corresponds to points near the "boundary" of the improvement region,
    # similar to how hypervolume is most sensitive to points near the Pareto front.
    
    beta = 0.5
    eps = 1e-9
    
    # Alternative: use a power transformation of EI to emphasize larger improvements
    # score = EI^gamma where gamma > 1
    # But EI can be zero or negative, so we need to be careful.
    
    # Better approach: combine EI with a UCB-like exploration term weighted by 
    # the hypervolume sensitivity.
    
    # Hypervolume sensitivity is highest where the marginal density of improvement is high.
    # This is captured by pdf(z) * (f_best - mu) / sigma = pdf(z) * z * sigma / sigma = z * pdf(z)
    
    # Let's use: score = EI + alpha * sigma * pdf(z)
    # The second term is the "uncertainty-weighted" contribution.
    
    alpha = 0.3
    
    score = ei + alpha * sigma * pdf
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
