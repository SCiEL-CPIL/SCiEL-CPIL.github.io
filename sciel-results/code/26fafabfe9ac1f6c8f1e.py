"""Cost-aware acquisition: model-and-optimize framework.
Fits a parametric cost-normalized UCB to the observed data, optimizing the
exploration parameter (kappa) to maximize predicted improvement per unit cost.
This replaces the fixed heuristic with an adaptive, data-driven exploration
constant that balances exploitation and exploration based on the current GP state.
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

    # Relative sigma floor to keep UCB meaningful when GP is very confident.
    mu_min = float(np.min(mu))
    span = max(f_best - mu_min, 1e-9)
    sigma_floor_frac = 0.05
    sigma_floor = max(1e-6, sigma_floor_frac * span)
    sigma_safe = np.maximum(sigma, sigma_floor)

    # Deadline guard: fall back to a minimal valid score if near the limit.
    if not _elapsed_ok():
        score = np.zeros_like(mu)
        return score

    # Model-and-optimize framework:
    # 1. Define a parametric family: score_k(x) = (f_best - mu(x) + k * sigma(x)) / cost(x)
    # 2. Optimize k (exploration parameter) to maximize the "predicted improvement per cost"
    #    We use a simple grid search over k in a reasonable range.
    
    # The optimal k should balance:
    # - High k: more exploration (higher scores in high-uncertainty regions)
    # - Low k: more exploitation (higher scores near current best)
    
    # We estimate the "quality" of a k by looking at the distribution of scores.
    # A good k should produce a clear separation between promising and non-promising points.
    
    # Grid of k values to test
    k_grid = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    
    # For each k, compute the score vector and a "quality" metric
    # Quality metric: we want the top scores to be high and the distribution to be spread out
    # (indicating the model is confident about which points are good).
    # We use the mean of the top 1% scores as a proxy for "best predicted improvement per cost".
    
    top_frac = 0.01  # top 1%
    n_top = max(1, int(len(mu) * top_frac))
    
    best_k = 2.0
    best_quality = -np.inf
    
    for k in k_grid:
        # Compute scores for this k
        ucb = (f_best - mu) + k * sigma_safe
        scores_k = ucb / cost_safe
        
        # Ensure finite
        scores_k = np.where(np.isfinite(scores_k), scores_k, 0.0)
        scores_k = np.maximum(scores_k, 0.0)
        
        # Quality metric: mean of top scores
        if len(scores_k) > n_top:
            top_scores = np.sort(scores_k)[-n_top:]
            quality = np.mean(top_scores)
        else:
            quality = np.mean(scores_k) if len(scores_k) > 0 else 0.0
        
        # Additional consideration: penalize if the score distribution is too flat
        # (indicating the model is not discriminating well)
        if len(scores_k) > 10:
            score_std = np.std(scores_k)
            score_mean = np.mean(scores_k)
            # If std is very small relative to mean, the model is not discriminating
            if score_mean > 1e-6:
                cv = score_std / score_mean
                if cv < 0.1:
                    quality *= 0.5  # penalize flat distributions
        
        if quality > best_quality:
            best_quality = quality
            best_k = k
    
    # Use the best k to compute final scores
    kappa = best_k
    ucb = (f_best - mu) + kappa * sigma_safe
    score = ucb / cost_safe

    # Numerical safety: finite, non-negative.
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)

    return score
# EVOLVE-BLOCK-END
