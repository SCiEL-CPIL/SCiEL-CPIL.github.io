# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using knowledge-gradient-inspired
    scoring with cost normalization.
    
    Parameters:
    - mu: array of shape (2000,) with GP mean predictions
    - sigma: array of shape (2000,) with GP standard deviation predictions
    - f_best: scalar, the lowest observed value so far
    - cost: array of shape (2000,) with evaluation costs for each candidate
    
    Returns:
    - scores: array of shape (2000,) with acquisition values (higher = evaluate next)
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-6)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Safety margin check
    if time.time() - t_start > 280.0:
        # Fallback: simple cost-normalized improvement
        score = (f_best - mu) / cost
        return np.where(np.isfinite(score), score, 0.0)
    
    # Knowledge-gradient-inspired score:
    # Estimate expected improvement in f_best if we evaluate x.
    # For a single evaluation, the KG is approximately:
    #   E[max(f_best, f(x))] - f_best
    # where f(x) ~ N(mu, sigma^2)
    
    # The expected improvement from evaluating x:
    # EI = E[max(0, f_best - f(x))]
    # But KG also accounts for the value of information beyond just improvement.
    
    # A practical KG approximation for one step:
    # 1. Direct improvement: EI(f_best, mu, sigma)
    # 2. Information value: how much sigma reduction helps future decisions
    #    Approximated by sigma * (1 - norm.cdf(|z|)) where z = (f_best - mu)/sigma
    
    # Compute standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Information value term: captures the benefit of reducing uncertainty
    # Higher when we're uncertain AND the point could be near the optimum
    # Use a soft indicator: points with low mu (potentially good) and high sigma (uncertain)
    info_value = sigma_safe * norm.pdf(z)
    
    # Combine: direct improvement + weighted information value
    # Weight the information term to balance exploration/exploitation
    info_weight = 0.3
    
    kg_score = ei + info_weight * info_value
    
    # Cost-aware normalization:
    # Instead of dividing by cost, use a cost penalty that allows
    # slightly higher-cost points if they have significantly better scores.
    # This is more flexible than strict 1/cost normalization.
    
    # Use sqrt(cost) normalization as a compromise:
    # - Cheaper points still favored (1/sqrt(cost) > 1/cost for cost > 1)
    # - But doesn't over-penalize moderately expensive points
    # - The inner 80% region has cost in [1, 2], so sqrt gives [1, ~1.41]
    #   vs cost gives [1, 2], allowing more exploration of the full space
    
    cost_norm = np.sqrt(cost)
    scores = kg_score / cost_norm
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
