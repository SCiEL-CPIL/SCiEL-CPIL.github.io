"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Per-process deadline guard
_T0 = time.time()
_DEADLINE = 330.0

def _elapsed():
    return time.time() - _T0

def acquisition(mu, sigma, f_best, cost):
    # Guard: if we are dangerously close to the wall-clock limit, return a
    # deterministic, finite, low-risk signal immediately.
    if _elapsed() > _DEADLINE - 1.0:
        return np.zeros(np.shape(mu), dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Sanitize inputs
    sigma = np.clip(sigma, 1e-6, None)
    cost = np.clip(cost, 1e-6, None)
    
    # Structural change: Use UCB (Upper Confidence Bound) instead of EI.
    # For minimization, we want to minimize the predicted value while
    # exploring uncertain regions. UCB for minimization:
    #   score = -(mu - kappa * sigma) = -mu + kappa * sigma
    # This linearly combines exploitation (-mu) and exploration (kappa * sigma).
    #
    # Key structural difference from EI:
    #   EI = (f_best - mu) * Phi(z) + sigma * phi(z)  [nonlinear, bounded]
    #   UCB = -mu + kappa * sigma                      [linear, unbounded]
    #
    # UCB responds more directly to high-uncertainty regions and avoids the
    # nonlinear saturation that EI exhibits when sigma is large relative to
    # (f_best - mu). This can lead to more effective exploration in early
    # iterations when the GP is uncertain.
    
    # Exploration weight: moderate for a 2D problem with limited budget
    kappa = 1.0
    
    # UCB score for minimization (higher is better for acquisition)
    ucb = -mu + kappa * sigma
    
    # Cost-aware score: UCB / cost
    # We want to maximize this score.
    score = ucb / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
