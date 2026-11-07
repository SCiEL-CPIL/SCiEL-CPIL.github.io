# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: track wall-clock to avoid being killed.
# The evaluator kills after 330s wall-clock. Each acquisition call should be fast.
# We use a simple time check to ensure we don't exceed a per-call budget.
_start_time = time.time()
# Per-call budget: we want to be well under any reasonable limit.
# The main loop does ~20 evaluations (cost 30 / ~1.5 avg cost), so 20 calls.
# 330s / 20 calls = 16.5s per call, but we should be much faster.
# Let's budget 5s per call to be safe, but the actual computation is O(N) and fast.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses adaptive UCB / cost with a structural change from EI.
    
    Parameters:
    - mu: mean of GP predictions, shape (2000,)
    - sigma: std dev of GP predictions, shape (2000,)
    - f_best: best observed function value
    - cost: cost of evaluating each candidate, shape (2000,)
    
    Returns:
    - scores: acquisition values, shape (2000,)
    """
    # Deadline guard: check elapsed time
    elapsed = time.time() - _start_time
    if elapsed > 325:  # 5s safety margin
        # Return a simple fallback score
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        scores = -mu / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Clip cost to avoid division by zero and negative costs
    cost_safe = np.maximum(cost, 1e-9)
    
    # Adaptive UCB: -mu + kappa * sigma
    # The key insight: for minimization, UCB = -mu + kappa*sigma
    # We want high scores for good candidates.
    # 
    # Adaptive kappa: as f_best approaches 0 (the minimum), we reduce exploration.
    # f_best starts around 0.5-2 and decreases to near 0.
    # kappa should be high when f_best is large (early, explore more)
    # and low when f_best is small (late, exploit more).
    #
    # A natural adaptation: kappa = base_kappa * (1 + f_best)
    # When f_best is large, kappa is larger (more exploration).
    # When f_best approaches 0, kappa approaches base_kappa.
    #
    # Alternatively, we can use a decay: kappa = base_kappa / (1 + f_best)
    # This gives high exploration early and low exploration late.
    #
    # Let's use: kappa = base_kappa * (1 + 1/f_best_clipped)
    # No, let's think more carefully.
    #
    # The measured improvement was 1 -> 1.086 with adaptive UCB/cost.
    # Let's use a similar approach but with a different structural choice.
    #
    # Structural change: use a combination of UCB and a cost-normalized
    # expected improvement, but weight them adaptively.
    #
    # Actually, let's go with a cleaner structural change:
    # Use "Knowledge Gradient" inspired scoring, but simplified.
    # Or use a "probability of improvement" with cost normalization.
    #
    # Let's try: score = (-mu + kappa * sigma) / cost
    # where kappa adapts based on the scale of sigma relative to f_best.
    
    # Compute z-score for reference
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Adaptive kappa: scale exploration based on how close f_best is to 0
    # When f_best is large, we can afford more exploration.
    # When f_best is small, we should exploit.
    # Use: kappa = base_kappa * (1 + f_best)
    # This means early on (f_best ~ 1), kappa ~ 2*base_kappa
    # Late (f_best ~ 0.01), kappa ~ 1.01*base_kappa
    
    base_kappa = 1.5
    kappa = base_kappa * (1.0 + max(f_best, 0.0))
    
    # UCB for minimization: -mu + kappa * sigma
    ucb = -mu + kappa * sigma_safe
    
    # Cost-aware UCB
    scores = ucb / cost_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
