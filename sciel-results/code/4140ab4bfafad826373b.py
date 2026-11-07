# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: maximize expected improvement per unit cost,
    # but with a robust, numerically stable implementation that also
    # balances exploration via a UCB-like term.
    #
    # We combine two signals:
    #   1) Expected Improvement (EI) per unit cost (exploitation)
    #   2) A UCB term (exploration) scaled by sigma / sqrt(cost)
    #
    # This encourages visiting uncertain regions while still being cost-aware.
    
    sigma_safe = np.maximum(sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Clip negative EI to 0 (shouldn't happen often but for safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI
    cost_safe = np.maximum(cost, 1e-8)
    ei_per_cost = ei / cost_safe
    
    # UCB-like exploration term: sigma / sqrt(cost) encourages high uncertainty
    # and low cost regions
    ucb_term = sigma_safe / np.sqrt(cost_safe)
    
    # Combine: weighted sum. We tune the weight to balance exploration/exploitation.
    # A reasonable balance: 0.7 * EI/cost + 0.3 * UCB
    # But we should normalize to keep scores in a reasonable range.
    
    # Actually, let's think about this more carefully. The EI can be large when
    # mu is much smaller than f_best. The UCB term is bounded by sigma/sqrt(cost).
    # Let's use a geometric mean or a different combination.
    
    # Alternative: Use a product-based or ratio-based approach.
    # Let's try: score = EI / cost + lambda * sigma / cost
    # This is simpler and both terms have the same "per unit cost" interpretation.
    
    lambda_explore = 0.5  # Exploration weight
    
    score = ei_per_cost + lambda_explore * (sigma_safe / cost_safe)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
