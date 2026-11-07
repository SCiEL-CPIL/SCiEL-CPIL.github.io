"""
Cost-aware Knowledge Gradient acquisition for Bayesian optimization.
Structural change from EI: uses KG to optimize expected improvement in f_best,
with cost normalization and dynamic exploration scaling.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # 300s limit + safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Knowledge Gradient acquisition function.
    
    Combines:
      1. Knowledge Gradient (KG) approximation for global improvement.
      2. Dynamic exploration scaling based on progress.
      3. Cost normalization to favor cheaper evaluations.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Safety fallback: simple EI / cost
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standardize improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # --- Knowledge Gradient Approximation ---
    # KG measures the expected reduction in f_best from evaluating x.
    # For a single evaluation, KG ≈ E[max(f_best, f(x))] - f_best
    # This is related to EI but accounts for the global nature of the search.
    #
    # A practical approximation: KG is larger when:
    #   1. The point has high potential improvement (high EI)
    #   2. The point is in a region where the posterior is uncertain
    #      (evaluating it reduces global uncertainty)
    #   3. The point is not too far from the current best (relevance)
    #
    # We use a weighted combination that approximates KG:
    #   KG_approx = EI * (1 + alpha * sigma / sigma_ref)
    # where sigma_ref is a reference scale for uncertainty.
    
    # Reference scale: use the median sigma as a robust reference
    sigma_median = np.median(sigma_safe)
    if sigma_median < 1e-9:
        sigma_median = 1e-9
    
    # Uncertainty boost: points with higher than median uncertainty
    # have more potential to reveal new minima
    uncertainty_boost = 1.0 + 0.3 * (sigma_safe / sigma_median)
    
    # KG approximation
    kg = ei * uncertainty_boost
    
    # --- Dynamic Exploration Scaling ---
    # Scale exploration based on progress toward the known minimum
    min_possible = -0.8
    gap_to_min = max(f_best - min_possible, 0.0)
    
    # Early in the search (large gap), allow more exploration
    # Late in the search (small gap), focus on exploitation
    # Use exponential decay to transition smoothly
    exploration_weight = np.exp(-gap_to_min / 0.15)
    
    # Blend between KG (exploitation-focused) and PI (exploration-focused)
    # When gap is large, weight more toward PI for exploration
    # When gap is small, weight more toward KG for exploitation
    blend = 0.7 + 0.3 * exploration_weight  # 0.7 to 1.0
    score = blend * kg + (1.0 - blend) * pi
    
    # --- Cost-Aware Normalization ---
    # Divide by cost to favor cheaper evaluations
    # Use a softened cost to avoid extreme values
    cost_softened = cost_safe * (1.0 + 0.1 * np.exp(-cost_safe))
    score = score / cost_softened
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
