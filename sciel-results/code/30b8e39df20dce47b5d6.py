# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Pareto-frontier filtering + UCB for Bayesian optimization.
    
    Strategy:
    1. Filter out candidates whose predicted mean is worse than f_best by more than a tolerance.
    2. Apply a pure UCB score to the remaining candidates.
    3. Return scores (higher = evaluate next) for all 2000 candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure valid sigma
    sigma = np.maximum(sigma, 1e-12)
    
    n = len(mu)
    if n == 0:
        return np.array([])
    
    # Deadline guard: track elapsed time
    t0 = time.time()
    
    # Step 1: Pareto-frontier filtering
    # Mask out candidates whose mu is worse than f_best by more than a tolerance.
    # We want to restrict to regions with plausible improvement.
    # Tolerance: allow some slack for exploration near the boundary.
    # A candidate is "plausible" if mu < f_best + tol, where tol scales with sigma.
    
    # Use a dynamic tolerance: tol = max(0.1 * sigma, 0.01 * f_best, 1e-4)
    # This allows candidates with slightly worse mean but high uncertainty.
    tol = np.maximum(0.1 * sigma, np.maximum(0.01 * abs(f_best), 1e-4))
    plausible = mu < (f_best + tol)
    
    # If too few candidates pass the filter, relax it
    n_plausible = np.sum(plausible)
    if n_plausible < max(10, n // 10):
        # Relax: use a larger tolerance
        tol_relaxed = np.maximum(0.5 * sigma, np.maximum(0.05 * abs(f_best), 1e-3))
        plausible = mu < (f_best + tol_relaxed)
        n_plausible = np.sum(plausible)
    
    if n_plausible < 1:
        # Fallback: no candidates pass, use all
        plausible = np.ones(n, dtype=bool)
    
    # Step 2: Apply UCB score to plausible candidates
    # UCB = -mu + beta * sigma
    # Choose beta adaptively based on f_best and the number of plausible candidates.
    
    # When f_best is close to 0 (good), we want to exploit more (lower beta).
    # When f_best is far from 0, we want to explore more (higher beta).
    # Also, if there are many plausible candidates, we can be more selective (lower beta).
    
    # Base beta: starts high for exploration, decreases as we improve.
    # f_best typically ranges from ~1.0 down to ~0.0.
    # Let's use: beta = 1.0 + 2.0 * exp(-f_best * 5)
    # When f_best = 0: beta = 3.0 (high exploration)
    # When f_best = 1: beta = 1.0 + 2*exp(-5) ≈ 1.007 (mostly exploitation)
    
    # Actually, let's think about it differently.
    # Early on: f_best is high, we want to explore -> high beta.
    # Late on: f_best is low, we want to exploit -> low beta.
    # But wait, if f_best is low (close to optimum), we should trust the GP mean more.
    
    # Let's use: beta = 0.5 + 1.5 * (1 - min(f_best, 1.0))
    # When f_best = 0: beta = 0.5 (exploit)
    # When f_best = 1: beta = 2.0 (explore)
    
    f_best_clamped = np.clip(f_best, 0.0, 1.0)
    beta = 0.5 + 1.5 * (1.0 - f_best_clamped)
    
    # Compute UCB scores for all candidates
    ucb = -mu + beta * sigma
    
    # Step 3: Set scores for non-plausible candidates to a very low value
    # This ensures they are never selected.
    scores = np.full(n, -1e6, dtype=np.float64)
    scores[plausible] = ucb[plausible]
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    # Deadline check: if we're running out of time, return immediately
    elapsed = time.time() - t0
    if elapsed > 280:  # Safety margin before 300s limit
        return scores
    
    return scores
# EVOLVE-BLOCK-END
