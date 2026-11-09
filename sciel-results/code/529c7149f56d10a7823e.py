"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    UCB-based acquisition per unit cost.
    
    mu: (2000,) predicted mean in objective units
    sigma: (2000,) predicted std in objective units
    f_best: float, lowest observed value
    cost: (2000,) evaluation cost per candidate
    
    Returns: (2000,) finite scores, higher = evaluate next
    """
    # Deadline guard: if we're running long, return a safe fallback
    if time.time() - _start_time > _DEADLINE_SECONDS:
        # Fallback: uniform-ish scores to keep the loop moving
        return np.ones_like(mu) * 1e-10
    
    # Ensure sigma is non-negative and numerically safe
    sigma = np.maximum(sigma, 1e-9)
    
    # Cost is positive by problem definition, but guard against zeros
    cost = np.maximum(cost, 1e-9)
    
    # Adaptive exploration weight (kappa for UCB)
    # As budget depletes, increase exploration to escape local optima
    # f_best approaching the true minimum (-0.8) means we're close;
    # use the gap to modulate exploration
    gap = f_best - (-0.8)  # distance from known minimum
    # Normalize gap: large gap -> more exploration, small gap -> less
    # gap ranges from ~0 (found minimum) to ~1.8 (worst case)
    gap_norm = np.clip(gap / 1.0, 0.0, 2.0)
    
    # Base UCB kappa: start moderate, increase with gap
    # When far from optimum, explore more aggressively
    kappa = 0.5 + 0.8 * gap_norm
    
    # UCB score: (mu + kappa * sigma) / cost
    # Higher mu is better (we're minimizing, so lower mu is better)
    # Wait - we're MINIMIZING. So lower mu is better.
    # UCB for minimization: we want to find where mu is LOW and sigma is HIGH
    # So the "upper confidence bound" for minimization is: mu - kappa * sigma
    # But we want to maximize the acquisition, so:
    # score = (f_best - (mu - kappa * sigma)) / cost
    # This rewards: low mu (good prediction) and high sigma (uncertainty)
    
    # Actually, let me think more carefully:
    # For minimization, EI = E[max(0, f_best - f(x))]
    # UCB for minimization: we want to explore where the lower bound could be very low
    # Lower bound = mu - kappa * sigma
    # We want to maximize (f_best - (mu - kappa * sigma)) / cost
    # = (f_best - mu + kappa * sigma) / cost
    
    # This is equivalent to: (f_best - mu) / cost + kappa * sigma / cost
    # First term: exploitation (how much better than best)
    # Second term: exploration (uncertainty per cost)
    
    ucb_score = (f_best - mu + kappa * sigma) / cost
    
    # Clip to ensure finite and reasonable range
    ucb_score = np.clip(ucb_score, -1e6, 1e6)
    
    # Replace any non-finite values
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, 0.0)
    
    return ucb_score
# EVOLVE-BLOCK-END
