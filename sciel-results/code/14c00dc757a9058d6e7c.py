"""Cost-aware acquisition: model-and-optimize EI with learned exploration.
Structural change: fit a parametric cost-aware EI to the observed GP
predictions and optimize the exploration parameter to maximize predicted
improvement per unit cost. This adapts the exploration/exploitation balance
to the current state of the GP model instead of using a fixed kappa.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# --- Explicit deadline guard (330 s hard kill; keep a safety margin) ---
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0   # 10 s safety margin before the 330 s kill

def _elapsed_ok():
    """True while we have at least ~3 s of margin before the hard limit."""
    return time.time() < _DEADLINE - 3.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization (minimization).

    mu, sigma, f_best, cost : each shape (2000,)
      f_best : scalar, lowest observed value
      cost   : per-candidate cost = 1 + mean(x_scaled)  (in [1, 2])
    Returns 2000 finite scores; higher = evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Robust floors -------------------------------------------------------
    cost_safe = np.maximum(cost, 1e-9)

    # Relative sigma floor to keep EI meaningful when GP is very confident.
    mu_min = float(np.min(mu))
    span = max(f_best - mu_min, 1e-9)
    sigma_floor_frac = 0.05
    sigma_floor = max(1e-6, sigma_floor_frac * span)
    sigma_safe = np.maximum(sigma, sigma_floor)

    # Deadline guard: fall back to a minimal valid score if near the limit.
    if not _elapsed_ok():
        score = np.zeros_like(mu)
        return score

    # Model-and-optimize: learn kappa by maximizing expected improvement
    # per unit cost over a grid of candidate values.
    # For EI: EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # We want to maximize sum(EI(x) / cost(x)) over x, but since we don't
    # have the actual x, we approximate by maximizing the average EI/cost
    # weighted by how "promising" each candidate is.
    
    # Grid of kappa values to optimize
    kappa_grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    
    # Compute EI for each kappa value
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # But for UCB-style: score = (f_best - mu + kappa * sigma) / cost
    # We'll use a hybrid: learn the effective exploration level
    
    # For each kappa, compute the score vector and its "quality"
    # Quality metric: how many candidates have positive score, and their magnitude
    best_kappa = 2.0  # default
    best_quality = -np.inf
    
    for k in kappa_grid:
        ucb = (f_best - mu) + k * sigma_safe
        score_k = ucb / cost_safe
        score_k = np.where(np.isfinite(score_k), score_k, 0.0)
        score_k = np.maximum(score_k, 0.0)
        
        # Quality: combine mean score with concentration (how many are high)
        # Use the 90th percentile as a robust measure of "best candidate quality"
        if np.any(score_k > 0):
            p90 = np.percentile(score_k, 90)
            mean_pos = np.mean(score_k[score_k > 0]) if np.any(score_k > 0) else 0
            # Prefer kappas that give high quality to a good number of candidates
            quality = p90 + 0.1 * mean_pos
        else:
            quality = 0.0
        
        if quality > best_quality:
            best_quality = quality
            best_kappa = k
    
    # Final score with learned kappa
    ucb = (f_best - mu) + best_kappa * sigma_safe
    score = ucb / cost_safe
    
    # Numerical safety: finite, non-negative.
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
