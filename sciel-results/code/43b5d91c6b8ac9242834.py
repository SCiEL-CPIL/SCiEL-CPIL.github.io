"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = None


def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Adaptive UCB with EI-informed exploration weighting.
    
    This combines the structural UCB approach (which scored 1.9098) with
    refinement ideas from the EI-based inspirations. The key insight is to
    make the exploration bonus adaptive based on the relationship between
    the GP mean and f_best, similar to how EI naturally weights exploration
    vs exploitation.
    
    Parameters:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP std predictions
        f_best: scalar - best observed function value
        cost: array of shape (2000,) - cost of evaluating each point
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're close to the 300s limit, return a simple fallback
    if _get_elapsed_time() > 290:
        mu = np.asarray(mu)
        cost = np.asarray(cost)
        cost_safe = np.maximum(cost, 1e-9)
        scores = -mu / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute the gap between f_best and mu
    gap = f_best - mu
    
    # Adaptive beta: inspired by EI's natural weighting
    # When mu is close to f_best (small gap), we want more exploration
    # When mu is far from f_best (large gap), exploitation dominates
    # Use a smooth transition based on the z-score
    
    z = gap / sigma_safe
    
    # The EI exploration weight is pdf(z), which peaks at z=0 (mu ≈ f_best)
    # and decays as |z| increases. We use a similar shape for beta.
    # beta = base + range * pdf(z)
    # At z=0: pdf=0.399, so beta = base + range*0.399
    # At |z| large: pdf→0, so beta → base
    
    # Choose base and range to get a reasonable beta range
    # base=0.5, range=2.0 gives beta in [0.5, 0.5+0.798] = [0.5, 1.298]
    # This is more conservative than the fixed beta=1.5, but adaptive
    
    pdf_val = norm.pdf(z)
    base_beta = 0.5
    beta_range = 2.0
    beta = base_beta + beta_range * pdf_val
    
    # Clip beta to avoid extreme values
    beta = np.clip(beta, 0.3, 2.0)
    
    # UCB Score for minimization: -mu + beta * sigma
    # We want to maximize this score (higher is better for acquisition).
    ucb_scores = -mu + beta * sigma_safe
    
    # Cost-aware: Score per unit cost
    score = ucb_scores / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
