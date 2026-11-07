"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a Knowledge Gradient (KG)-inspired exploration term.
    
    The KG term approximates the expected reduction in the optimal value
    by considering how much evaluating a point could lower the current best.
    We use a practical approximation: the probability that the true value
    at x is below f_best, weighted by the expected improvement magnitude,
    divided by cost.
    
    Additionally, we add a small uncertainty-driven exploration bonus to
    encourage visiting under-explored regions, which helps escape local
    optima in the Ackley function's landscape.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Knowledge Gradient approximation:
    # P(true value < f_best) * expected improvement magnitude
    # This emphasizes points where we're uncertain AND the improvement could be large
    prob_improve = norm.cdf(z)
    kg_approx = prob_improve * ei
    
    # Exploration bonus: encourage visiting points with high uncertainty
    # relative to the range of observed values. This helps with the Ackley
    # function's multimodal structure.
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-10:
        exploration_bonus = 0.1 * (sigma_safe / sigma_max)
    else:
        exploration_bonus = 0.0
    
    # Combine: primarily cost-aware EI, with KG weighting and exploration
    # The KG term down-weights points that are very likely to be good
    # (high prob_improve) unless the improvement is also large,
    # which can help balance exploitation vs. exploration.
    score = (ei + 0.5 * kg_approx + exploration_bonus) / np.maximum(cost, 1e-8)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
