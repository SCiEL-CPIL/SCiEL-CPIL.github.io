# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: the evaluator kills after 330s wall-clock.
# We keep a safety margin to return valid results before the limit.
_DEADLINE = 330.0 - 30.0  # 300s safety margin
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _within_budget():
    return _elapsed() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a budget-decaying UCB with cost normalization:
    - Early on (low cost): high exploration bonus (beta large)
    - Late on (high cost): shift to exploitation (beta small)
    - Cost normalization: prefer cheap points
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Handle edge cases
    if n == 0:
        return np.array([])
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Budget-decaying UCB parameter
    # Total budget is 30, starting cost is ~5 (5 uniform points)
    # beta decays from ~2.5 to ~0.5 as cost approaches budget
    current_cost = float(np.mean(cost)) if len(cost) > 0 else 5.0
    # Normalize: cost goes from ~5 to 30
    cost_frac = np.clip((current_cost - 5.0) / 25.0, 0.0, 1.0)
    # beta: high exploration early, low late
    beta = 2.5 - 2.0 * cost_frac
    
    # UCB: mu + beta * sigma
    ucb = mu + beta * sigma_safe
    
    # Cost normalization: divide by cost to prefer cheaper evaluations
    # But we want higher score = better, and lower cost = better
    # So score = ucb / cost (since we're minimizing, lower mu is better)
    # Wait - for minimization, we want low mu. UCB for minimization:
    # We want points with low mu (good) but also high uncertainty (sigma)
    # For minimization, the "upper confidence bound" on the objective
    # is mu + beta*sigma (worst case), but we want to minimize, so
    # we actually want to maximize the expected improvement or minimize
    # the lower confidence bound.
    
    # Let me reconsider. For minimization:
    # - mu is the predicted mean of f(x) (lower is better)
    # - sigma is the standard deviation
    # - We want to find low values of f
    # - LCB (Lower Confidence Bound) for minimization: mu - beta*sigma
    #   This gives a lower bound on what f(x) could be (optimistic)
    # - We want to maximize the "optimism" which means minimizing LCB
    # - So acquisition = -LCB = -mu + beta*sigma, then normalize by cost
    
    # Actually, let's use the standard approach:
    # For minimization, expected improvement is:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # This is positive when mu < f_best (improvement expected)
    
    # But the task says "higher = evaluate next", so we want high scores for good candidates.
    # For minimization, good candidates have low mu and/or high sigma (uncertainty).
    
    # Let's use: score = EI / cost, where EI is expected improvement for minimization
    # This balances improvement potential against evaluation cost.
    
    # Expected Improvement for minimization:
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Add a small UCB component for exploration
    # ucb_component = beta * sigma_safe
    # combined = ei + ucb_component
    # But let's keep it simple: just EI / cost with the budget-decaying beta
    # incorporated via a modified EI.
    
    # Actually, let's use a hybrid: EI + beta * sigma, all divided by cost
    # This gives exploration bonus scaled by cost
    score = (ei + beta * sigma_safe) / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
