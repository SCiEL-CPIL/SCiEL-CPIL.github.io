"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIMEOUT = 290  # Safety margin before 300s


def acquisition(mu, sigma, f_best):
    """
    Quantile-based Expected Improvement (QEI) acquisition function.
    Targets the 95th percentile of the posterior predictive distribution for robust
    exploitation in the multimodal Styblinski-Tang landscape.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean of the GP at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation of the GP at candidate points.
    f_best : float
        Best observed function value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Check deadline - if we're running out of time, return a fast fallback
    if time.time() - _START_TIME > _TIMEOUT:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)

    # Quantile-based Expected Improvement:
    # We compute the expected improvement at a quantile of the posterior.
    # For minimization, the improvement is max(0, f_best - f(x)).
    # The 95th percentile quantile approach: we look at the distribution of f(x) | X
    # and consider the improvement relative to f_best at a high confidence level.
    #
    # The posterior predictive at each point is Normal(mu, sigma^2).
    # The CDF of improvement I = max(0, f_best - f) can be computed.
    # 
    # A more direct approach for QEI:
    # Let z = (f_best - mu) / sigma
    # The expected improvement is E[I] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # For quantile-based EI, we can use a modified z that accounts for the quantile.
    #
    # Alternative: Use the fact that the 95th percentile of the posterior is mu - 1.645*sigma
    # (for minimization, lower is better). The improvement at this quantile is:
    # I_95 = max(0, f_best - (mu - 1.645*sigma)) = max(0, f_best - mu + 1.645*sigma)
    #
    # But a better QEI formulation uses the expected improvement conditional on
    # the posterior being in the lower 5% (high confidence improvement region).
    #
    # Let's use a simpler, effective approach:
    # QEI = E[max(0, f_best - f)] where f ~ N(mu, sigma^2), but weighted toward
    # high-confidence regions by using a modified z-score.
    
    # Standard EI for reference
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Quantile-based approach: Use the 95th percentile of the predictive distribution
    # For minimization, the 95th percentile (lower tail) is mu - 1.645 * sigma
    # The improvement at this quantile gives a conservative (high-confidence) estimate
    z_quantile = norm.ppf(0.95)  # 1.645 for 95th percentile
    
    # The expected improvement considering the quantile:
    # We compute E[max(0, f_best - (mu - z_q * sigma))]
    # = E[max(0, (f_best - mu) + z_q * sigma)]
    # Let d = f_best - mu + z_q * sigma
    # E[max(0, d)] where d is deterministic given mu, sigma
    # But this is just max(0, f_best - mu + z_q * sigma)
    
    # A more principled QEI: use the expected improvement at the alpha-quantile
    # This is equivalent to adjusting the z-score in the EI formula
    # QEI_alpha = (f_best - mu) * Phi(z) + sigma * phi(z) where we shift z
    
    # Let's use a hybrid: standard EI but with an exploration bonus that
    # scales with the quantile of the posterior
    alpha = 0.95  # 95th percentile
    z_alpha = norm.ppf(alpha)  # ~1.645
    
    # Quantile-based EI: We want to favor points where the lower 5% of the
    # predictive distribution is below f_best. This means:
    # P(f < f_best) > 0.95, i.e., (f_best - mu)/sigma > z_alpha
    # The "quantile improvement" can be formulated as:
    # QEI = (f_best - mu) * Phi(z) + sigma * phi(z) + bonus * sigma * I[z > z_alpha]
    
    # Actually, let's use a cleaner formulation:
    # The alpha-quantile of the improvement distribution
    # For f ~ N(mu, sigma^2), the CDF of I = max(0, f_best - f) is:
    # P(I <= i) = P(f_best - f <= i) = P(f >= f_best - i) = 1 - Phi((f_best - i - mu)/sigma)
    # The alpha-quantile of I is the value i_alpha such that P(I <= i_alpha) = alpha
    # This gives: 1 - Phi((f_best - i_alpha - mu)/sigma) = alpha
    # Phi((f_best - i_alpha - mu)/sigma) = 1 - alpha
    # (f_best - i_alpha - mu)/sigma = norm.ppf(1 - alpha) = -z_alpha
    # i_alpha = f_best - mu + z_alpha * sigma
    
    # So the 95th percentile of the improvement distribution is:
    # i_95 = max(0, f_best - mu + 1.645 * sigma)
    
    i_quantile = np.maximum(0, f_best - mu + z_alpha * sigma_safe)
    
    # Blend standard EI and quantile-based improvement
    # Standard EI captures the mean improvement
    # Quantile-based captures high-confidence improvement
    # Use a weighted combination
    
    # Normalize both to comparable scales
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = max(ei_max - ei_min, 1e-9)
    ei_norm = (ei - ei_min) / ei_range
    
    iq_min = np.min(i_quantile)
    iq_max = np.max(i_quantile)
    iq_range = max(iq_max - iq_min, 1e-9)
    iq_norm = (i_quantile - iq_min) / iq_range
    
    # Weight: favor quantile-based more when uncertainty is high (exploration)
    sigma_mean = np.mean(sigma_safe)
    sigma_weight = np.clip(sigma_safe / (sigma_mean + 1e-9), 0.0, 1.0)
    
    # Blend coefficient: 0.4 to 0.6, higher favors quantile (more conservative)
    blend_w = 0.4 + 0.2 * sigma_weight
    
    scores = blend_w * ei_norm + (1 - blend_w) * iq_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
