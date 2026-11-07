# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-entropy heuristic: maximize the ratio of expected information gain to
    evaluation cost, approximating the mutual information between a candidate's
    objective value and the posterior over the global minimum.

    For a single Gaussian observation y ~ N(mu_i, sigma_i^2), the expected
    reduction in posterior variance of the global minimum is proportional to
    sigma_i^2 * (1 - rho^2) where rho is the correlation between the observation
    and the global minimum. A practical surrogate for the mutual information
    between y and the global minimum is:

        MI_proxy = sigma_i^2 * (1 - exp(-(f_best - mu_i)^2 / (2 * sigma_i^2)))

    This captures:
      - High sigma_i: more information potential
      - Low mu_i relative to f_best: observation is likely to be informative
        about whether a new minimum exists

    The cost-aware score is:
        score = MI_proxy / (cost^alpha)

    where alpha > 1 penalizes expensive regions super-linearly, reflecting the
    idea that information gain per unit cost diminishes in expensive regions.

    We also add a small Expected Improvement term to ensure exploitation of
    known promising regions, weighted by the same cost penalty.
    """
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement (standard)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Mutual information proxy: expected information gain about global minimum
    # The term (1 - exp(-(f_best - mu)^2 / (2 * sigma^2))) is 0 when mu = f_best
    # (no new information) and approaches 1 when mu << f_best (high information)
    # Multiplied by sigma^2 to weight by uncertainty
    improvement_margin = f_best - mu
    mi_proxy = sigma_safe ** 2 * (1.0 - np.exp(-(improvement_margin ** 2) / (2.0 * sigma_safe ** 2)))
    
    # Cost penalty: super-linear to strongly penalize expensive regions
    # cost is in [1, 2] since x scaled to [0,1], so cost^1.5 gives range [1, ~2.83]
    cost_penalty = np.power(cost + eps, 1.5)
    
    # Combine: MI proxy dominates for exploration, EI provides exploitation
    # Weight EI by a small factor to balance
    score = (mi_proxy + 0.3 * ei) / cost_penalty
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
