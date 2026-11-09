"""Cost-aware acquisition function for Bayesian optimization (4 test functions)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_T0 = time.monotonic()
_DEADLINE = 325.0  # safety margin before 330s kill


def _elapsed():
    return time.monotonic() - _T0


def _deadline_ok():
    return _elapsed() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Adaptive Cost-Normalized Expected Improvement (AC-EI).
    
    Structural change from current program:
    - Current: EI / cost (fixed gamma=1)
    - New: EI / cost^gamma where gamma is learned from the data
    
    The gamma parameter controls how aggressively we penalize high-cost candidates.
    We estimate gamma by finding the value that maximizes the "efficiency" of the
    top-k most promising candidates, balancing exploitation (high EI) against 
    cost awareness.
    
    Additionally, we blend with a UCB-like exploration term to handle early-stage
    uncertainty better, with the blend weight also adaptive.
    """
    # Deadline guard: if we're running out of time, return a simple, fast score
    if not _deadline_ok():
        return -(mu + 0.5 * cost)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Defensive: ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Avoid division by zero and negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-6)

    # Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative

    # UCB for minimization: f_best - mu + beta * sigma
    # This captures exploration potential
    beta = 2.0
    ucb = f_best - mu + beta * sigma_safe
    ucb = np.maximum(ucb, 0.0)

    # Learn gamma: the exponent in cost^gamma
    # Strategy: try a small grid of gamma values and pick the one that
    # maximizes the "quality" of the selected top candidate.
    # Quality proxy: For each gamma, compute scores = ei / cost^gamma,
    # find the top-10 candidates, and average their (ei / cost) ratio.
    # We want gamma that makes the top candidates have good ei/cost.
    
    # Also learn alpha: blend between EI-based and UCB-based scores
    # Score = alpha * (ei / cost^gamma) + (1-alpha) * (ucb / cost^gamma)
    
    # To keep it fast, we use a small grid and vectorized operations
    gamma_grid = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    alpha_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Precompute log(cost) for efficient gamma application
    log_cost = np.log(cost_safe)
    
    # For each (gamma, alpha) pair, compute the score of the best candidate
    # and its ei/cost ratio. We want to maximize this ratio.
    
    # Vectorized approach:
    # For each gamma: cost^gamma = exp(gamma * log_cost)
    # For each alpha: score = alpha * ei + (1-alpha) * ucb, then divide by cost^gamma
    
    best_gamma = 1.0
    best_alpha = 0.5
    best_quality = -np.inf
    
    # Limit the grid search to keep it fast
    # We'll evaluate all combinations but use efficient numpy operations
    for gamma in gamma_grid:
        cost_power = np.exp(gamma * log_cost)
        inv_cost = 1.0 / cost_power
        
        for alpha in alpha_grid:
            # Combined metric before cost normalization
            combined = alpha * ei + (1.0 - alpha) * ucb
            scores = combined * inv_cost
            
            # Find top-10 candidates by score
            # Use argsort for top-k
            top_k = 10
            if len(scores) > top_k:
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                top_scores = scores[top_indices]
                top_ei = ei[top_indices]
                top_cost = cost_safe[top_indices]
            else:
                top_indices = np.arange(len(scores))
                top_scores = scores[top_indices]
                top_ei = ei[top_indices]
                top_cost = cost_safe[top_indices]
            
            # Quality: average of ei/cost for top candidates, weighted by their score
            # This rewards gammas that put high-ei/cost points at the top
            top_ei_over_cost = top_ei / top_cost
            # Weight by normalized score
            max_top_score = np.max(top_scores) if len(top_scores) > 0 else 1.0
            weights = top_scores / (max_top_score + 1e-9)
            quality = np.sum(weights * top_ei_over_cost) / (np.sum(weights) + 1e-9)
            
            if quality > best_quality:
                best_quality = quality
                best_gamma = gamma
                best_alpha = alpha

    # Compute final scores with learned parameters
    cost_power = np.exp(best_gamma * np.log(cost_safe))
    inv_cost = 1.0 / cost_power
    
    combined = best_alpha * ei + (1.0 - best_alpha) * ucb
    score = combined * inv_cost

    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
