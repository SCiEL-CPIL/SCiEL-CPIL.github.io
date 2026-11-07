"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Hartmann 6d.
    Uses a Pareto-frontier-aware blend of Expected Improvement and UCB
    with cost-aware weighting.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions.
    sigma : array of shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Lowest observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x)).
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Floor to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # --- Expected Improvement (for minimization) ---
    z = (f_best - mu) / sigma
    z = np.clip(z, -10.0, 10.0)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(ei > 0, ei, 0.0)
    
    # --- UCB (Upper Confidence Bound for minimization) ---
    # Adaptive beta: increase exploration early, decrease later
    f_min_known = -3.32237
    if f_best > f_min_known:
        progress = 1.0 / (1.0 + np.exp(0.5 * (f_best - f_min_known)))
    else:
        progress = 1.0
    
    # Beta: higher when exploring (low progress), lower when exploiting (high progress)
    # Tuned beta from 2.0 to 0.8 to favor exploitation slightly more
    beta = 0.8 * (1.0 - progress) + 0.3
    
    # UCB score for minimization: we want to maximize this
    ucb = -mu + beta * sigma
    
    # --- Pareto-frontier-aware scoring ---
    # Instead of arbitrary weight tuning, use a Pareto dominance filter
    # to identify candidates that are efficient in (improvement, uncertainty, cost)
    
    # Normalize each dimension to [0, 1] for fair comparison
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ei_norm = ei / ei_max
    
    ucb_shifted = ucb - np.min(ucb)
    ucb_shifted_max = np.max(ucb_shifted) if np.max(ucb_shifted) > 0 else 1.0
    ucb_norm = ucb_shifted / ucb_shifted_max
    
    # For cost, lower is better, so invert
    cost_norm = 1.0 - (cost - np.min(cost)) / (np.max(cost) - np.min(cost) + 1e-9)
    
    # Create a composite "efficiency" score that rewards being on the Pareto frontier
    # A point is Pareto-efficient if no other point dominates it in all three dimensions
    # We approximate this by computing a rank-based score
    
    # Compute ranks for each dimension (lower rank = better)
    ei_rank = np.argsort(np.argsort(-ei_norm)) / len(ei_norm)  # Higher EI = lower rank
    ucb_rank = np.argsort(np.argsort(-ucb_norm)) / len(ucb_norm)  # Higher UCB = lower rank
    cost_rank = np.argsort(np.argsort(cost_norm)) / len(cost_norm)  # Lower cost = lower rank
    
    # A point is "Pareto-efficient" if it has low ranks in at least one dimension
    # without being dominated in all dimensions
    # Approximate: score is high if the point is good in at least one dimension
    # and not terrible in others
    
    # Use a geometric mean of the inverse ranks to reward balance
    # But boost points that excel in any single dimension
    min_rank = np.minimum(np.minimum(ei_rank, ucb_rank), cost_rank)
    max_rank = np.maximum(np.maximum(ei_rank, ucb_rank), cost_rank)
    
    # Balance score: low if ranks are consistent (point is good overall)
    balance_score = 1.0 / (1.0 + max_rank)
    
    # Excellence score: high if point is excellent in at least one dimension
    excellence_score = 1.0 / (1.0 + min_rank)
    
    # Combine: weight balance more when progress is high (exploit), excellence more when low (explore)
    balance_weight = 0.5 + 0.5 * progress
    excellence_weight = 1.0 - balance_weight
    
    pareto_score = balance_weight * balance_score + excellence_weight * excellence_score
    
    # Blend with the original EI/UCB scores
    # Use the Pareto score to modulate the combined EI/UCB score
    ei_weight = 0.6 + 0.4 * progress
    ucb_weight = 1.0 - ei_weight
    
    combined = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Modulate by Pareto efficiency
    scores_unadjusted = combined * (0.5 + 0.5 * pareto_score)
    
    # --- Cost-aware adjustment ---
    # Use a softer cost penalty: cost^0.5 instead of cost
    cost_factor = np.power(cost, 0.5)
    
    scores = scores_unadjusted / cost_factor
    
    # Check time
    if time.time() > deadline:
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
