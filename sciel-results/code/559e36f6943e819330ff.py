# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best):
    """
    Multi-point Knowledge Gradient (KG) acquisition function.
    
    The true KG for selecting the next evaluation point x_i considers the
    expected reduction in the best-known value after evaluating x_i:
        KG_i = E[f_best - min(f(x_i), f_best)]
    
    For a single point, this equals EI. However, in a multi-modal landscape
    like Weierstrass, the key insight is that we should consider the
    probability that x_i reveals a NEW basin with a lower minimum than
    all previously explored basins.
    
    I'll implement an approximation to the full KG by considering:
    1. The direct EI (probability of improving on f_best at x_i)
    2. A "novelty" term that rewards points in regions where the GP
       predicts high variance AND the mean is not already well-explored
    
    For the novelty term, I use the fact that if sigma_i is high and
    mu_i is significantly above f_best, there's a chance that the true
    minimum in that region is below f_best. The probability that the
    true minimum in a neighborhood of x_i is below f_best can be
    approximated by considering the lower tail of the GP posterior.
    
    Specifically, for a point x_i, the probability that there exists
    a point in its "influence region" with value below f_best is
    approximately P(min_{x ~ N(x_i, r^2)} f(x) < f_best)
    
    For a 1D GP, this can be approximated using the variance of the
    GP at x_i and the correlation length.
    """
    if _elapsed() > _DEADLINE - _MARGIN:
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei
    
    n = len(mu)
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Standard EI (single-point KG)
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # KG extension: account for the multi-modal structure
    # 
    # In a multi-modal function, evaluating a point with high sigma
    # in a region where mu >> f_best is valuable because it may reveal
    # a new basin with a lower minimum. The value of such a point is
    # not just the probability of improving at that exact point, but
    # the expected reduction in the best value considering that the
    # true minimum in that region could be much lower.
    #
    # Approximation: For each point i, estimate the expected minimum
    # value in a local neighborhood of size ~ length_scale. The GP
    # variance at x_i gives us the uncertainty, and the Matern-5/2
    # kernel with length_scale=0.2 means that points within ~0.2
    # of x_i are correlated.
    #
    # The expected minimum of a GP over a region can be approximated
    # by mu_i - c * sigma_i, where c depends on the number of
    # correlated points. For a 1D problem with ~10 correlated points,
    # c ~ 2-3.
    #
    # So the "effective" best value we might find near x_i is:
    #   effective_best_i = mu_i - c * sigma_i
    #
    # The KG then becomes:
    #   KG_i = E[max(f_best - f(x_i), 0)] + bonus for the chance
    #         that the true minimum in the region is below f_best
    #
    # Let me use a different approach: compute the probability that
    # the minimum in a neighborhood of x_i is below f_best, weighted
    # by the expected improvement.
    
    # Estimate the expected minimum in a local region
    # Using the fact that for N correlated Gaussian samples with
    # standard deviation sigma, the expected minimum is approximately
    # mu - sigma * sqrt(2 * log(N))
    # For ~10 correlated points: sqrt(2*log(10)) ~ 2.15
    
    c_min = 2.0  # conservative estimate for the expected minimum offset
    
    # The "promised" best value if we explore around x_i
    promised_best = mu - c_min * sigma_safe
    
    # The probability that the true minimum in the region is below f_best
    # P(min < f_best) ≈ P(mu - c*sigma < f_best) = P(sigma > (mu - f_best)/c)
    # For each point, this is:
    #   P_i = 1 - Phi((mu_i - f_best) / (c * sigma_i))
    #        = Phi((f_best - mu_i) / (c * sigma_i))
    
    z_region = (f_best - mu) / (c_min * sigma_safe)
    prob_region_below = norm.cdf(z_region)
    
    # The expected improvement if the region minimum is below f_best
    # is approximately max(f_best - promised_best, 0)
    expected_improvement_region = np.maximum(f_best - promised_best, 0.0)
    
    # KG score: combine direct EI with the regional exploration value
    # The regional value is the probability of finding a better minimum
    # in the region, times the expected improvement
    regional_value = prob_region_below * expected_improvement_region
    
    # Weighted combination: direct EI + regional exploration
    # Use a weight that balances immediate improvement vs. long-term exploration
    alpha = 0.5
    
    scores = alpha * ei + (1 - alpha) * regional_value
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
