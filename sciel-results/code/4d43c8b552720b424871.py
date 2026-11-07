"""Rewritten program with a cost-aware, uncertainty-driven acquisition for Powell 4d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: keep a global start time and a safety margin.
_START_TIME = time.time()
DEADLINE = 300.0  # seconds; evaluator kills after 330s, use 300 for safety
SAFETY_MARGIN = 5.0  # seconds before deadline to stop heavy work


def _elapsed():
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function returning 2000 finite scores.
    
    Strategy: combine expected improvement (EI) with an uncertainty bonus,
    normalized by cost, and add a small exploration term to avoid over-exploitation.
    
    mu: (2000,) predicted mean
    sigma: (2000,) predicted std dev (clipped to min 1e-8)
    f_best: scalar, best observed value
    cost: (2000,) evaluation cost per candidate
    """
    # Check deadline; if close, return a safe default (uniform random-ish scores)
    if _elapsed() > DEADLINE - SAFETY_MARGIN:
        # Return finite scores that favor lower-cost points slightly
        return 1.0 / (cost + 1e-9)
    
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in norm.cdf/pdf
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI is non-negative; ensure finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Uncertainty bonus: reward high sigma (exploration)
    # Scale by f_best magnitude to keep balance
    scale = max(abs(f_best), 1e-6)
    uncertainty = sigma_safe / scale
    
    # Cost-aware: divide by cost
    cost_safe = np.maximum(cost, 1e-9)
    
    # Combined score: EI/cost + weight * uncertainty/cost
    # The weight balances exploitation vs exploration
    weight = 0.5
    score = (ei + weight * uncertainty) / cost_safe
    
    # Add a tiny random perturbation to break ties and encourage diversity
    # Use deterministic seed based on index to keep it reproducible within a call
    # Actually, just add a very small noise to avoid exact ties
    rng = np.random.default_rng(42)
    score += rng.random(len(score)) * 1e-10
    
    # Ensure all finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
