"""Rewritten program: structural change to a quantile-mixed cost-aware acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 300.0
_TIMEOUT_MARGIN = 20.0

def _check_timeout():
    return (time.time() - _START_TIME) > (_DEADLINE - _TIMEOUT_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Rastrigin 2D minimization.
    
    Structural change: replace the Pareto-frontier / dominance mechanism with
    a quantile-mixed probabilistic acquisition. We compute three scalar metrics
    - Expected Improvement (EI)
    - Probability of Improvement (PoI)
    - Lower Confidence Bound (LCB)
    then normalize each to [0,1] by its empirical distribution over the 2000
    candidates, mix them with fixed weights, and divide by cost.
    
    This is a different representation of the trade-off: instead of ranking
    points by multi-objective dominance, we compress each metric to its
    relative standing and form a single convex combination. The convex
    combination is cheap, robust to scale differences, and avoids the O(n^2)
    dominance test.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    if _check_timeout():
        # Fallback: simple normalized LCB divided by cost
        sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
        cost_safe = np.where(cost > 1e-8, cost, 1e-8)
        lcb = mu - 2.0 * sigma_safe
        score = -lcb / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    cost_safe = np.where(cost > 1e-8, cost, 1e-8)
    
    # z = (f_best - mu) / sigma  (improvement z-score)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -15, 15)
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    from scipy.stats import norm
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.where(ei > 0, ei, 0.0)
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    poi = Phi_z
    
    # Lower Confidence Bound (exploration):
    lcb = mu - 2.0 * sigma_safe  # negative = more promising
    # Convert to "higher is better": neg_lcb = -lcb
    neg_lcb = -lcb
    
    # Normalize each metric to [0, 1] via rank-based quantiles.
    # This makes the mixture robust to outliers and absolute scale.
    def _rank_normalize(arr):
        """Map arr to [0,1] using its rank among itself (1-based rank / n)."""
        order = np.argsort(arr, kind='mergesort')
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
        n = len(arr)
        return (ranks - 1.0) / max(n - 1, 1)
    
    ei_norm = _rank_normalize(ei)
    poi_norm = _rank_normalize(poi)
    lcb_norm = _rank_normalize(neg_lcb)
    
    # Fixed convex combination. Weights chosen to balance exploitation (EI),
    # probability signal (PoI), and exploration (LCB).
    w_ei = 0.50
    w_poi = 0.25
    w_lcb = 0.25
    
    blended = w_ei * ei_norm + w_poi * poi_norm + w_lcb * lcb_norm
    
    # Cost-aware: divide by cost (cheaper evaluations get higher score)
    score = blended / cost_safe
    
    # Guard against non-finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
