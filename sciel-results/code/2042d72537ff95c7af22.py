"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 300.0  # seconds, safety margin before 330s kill
_TIMEOUT_MARGIN = 25.0  # return at least 25s before deadline

def _check_timeout():
    return (time.time() - _START_TIME) > (_DEADLINE - _TIMEOUT_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (Rastrigin 2D).
    
    Strategy: Pareto-frontier selection over two independent metrics:
    - Improvement potential: (f_best - mu) / sigma  (z-score of improvement)
    - Cost efficiency: sigma / cost (uncertainty per unit cost)
    
    Non-dominated points are strongly boosted to preserve diversity for
    robust exploration of the multimodal Rastrigin landscape.
    """
    if _check_timeout():
        # Fallback: simple cost-aware exploitation
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost_safe = np.maximum(cost, 1e-9)
        return -mu / cost_safe

    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: non-positive sigma and cost
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    cost_safe = np.where(cost > 1e-8, cost, 1e-8)
    
    n = len(mu)
    
    # Metric 1: Improvement potential (z-score)
    # Higher means more likely to improve over f_best
    imp = (f_best - mu) / sigma_safe
    imp = np.clip(imp, -10, 10)
    
    # Metric 2: Cost efficiency
    # Higher means more uncertainty per unit cost (better exploration value)
    ce = sigma_safe / cost_safe
    
    # Normalize both metrics to [0, 1] for Pareto comparison
    imp_min, imp_max = np.min(imp), np.max(imp)
    ce_min, ce_max = np.min(ce), np.max(ce)
    
    imp_range = imp_max - imp_min if imp_max > imp_min else 1.0
    ce_range = ce_max - ce_min if ce_max > ce_min else 1.0
    
    imp_norm = (imp - imp_min) / imp_range
    ce_norm = (ce - ce_min) / ce_range
    
    # Compute Pareto dominance
    # Point i dominates point j if imp_i >= imp_j AND ce_i >= ce_j AND at least one strict
    imp_cmp = imp_norm[:, None] >= imp_norm[None, :]
    ce_cmp = ce_norm[:, None] >= ce_norm[None, :]
    
    imp_strict = imp_norm[:, None] > imp_norm[None, :]
    ce_strict = ce_norm[:, None] > ce_norm[None, :]
    
    dominates = (imp_cmp & ce_cmp) & (imp_strict | ce_strict)
    
    # Number of points that dominate each point
    n_dominating = np.sum(dominates, axis=0)
    
    # Non-dominated points have n_dominating == 0
    is_nondominated = (n_dominating == 0)
    
    # Base score: weighted sum of normalized metrics
    base_score = 0.5 * imp_norm + 0.5 * ce_norm
    
    # Boost non-dominated points strongly
    # Penalty for being dominated
    domination_penalty = n_dominating / n
    
    score = base_score + 0.8 * is_nondominated - 0.2 * domination_penalty
    
    # Final cost-aware scaling: prefer cheaper evaluations
    score = score / np.sqrt(cost_safe)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
