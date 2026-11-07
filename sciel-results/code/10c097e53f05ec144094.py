"""
Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D).

Structural change: Use a Pareto-frontier based acquisition that scores candidates
by their rank in a non-dominated sort of (predicted value, evaluation cost).
The predicted value combines exploitation (mu) and exploration (sigma) via UCB,
and the cost is the evaluation cost. Candidates on the cost-efficient frontier
(where marginal reduction in predicted value per unit cost is maximized) receive
the highest scores.

This is a different modelling family: instead of a single scalarized blend with
fixed or adaptive hyperparameters, we use a multi-objective ranking that naturally
balances the trade-off between predicted improvement and evaluation cost without
tuning a single weight parameter.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: keep a safety margin before the 330 s wall-clock kill
_DEADLINE = time.time() + 315.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.

    mu, sigma: shape (2000,) - GP mean and std (in objective units)
    f_best: scalar - lowest observed value
    cost: shape (2000,) - evaluation cost for each candidate

    Returns: shape (2000,) - scores (higher = evaluate next)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros(len(mu), dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = len(mu)

    # Guard against zero/negative values
    cost = np.maximum(cost, 1e-9)
    sigma = np.maximum(sigma, 1e-9)

    # --- UCB for minimization: lower predicted value is better ---
    # Use beta = 2.0 for a balanced exploration-exploitation trade-off
    # This is inspired by the top-performing solution
    beta = 2.0
    predicted_value = mu - beta * sigma  # Lower is better for minimization

    # --- Pareto-frontier based scoring ---
    # We have two objectives:
    #   1. Minimize predicted_value (lower is better)
    #   2. Minimize cost (lower is better)
    #
    # A point is on the Pareto frontier if no other point has both
    # lower (or equal) predicted_value AND lower (or equal) cost,
    # with at least one being strictly lower.
    #
    # For efficiency, we compute a rank-based score:
    #   - Sort by predicted_value
    #   - For each point, find the minimum cost among all points with
    #     lower or equal predicted_value
    #   - A point is on the frontier if its cost is less than or equal
    #     to the minimum cost of all points with strictly lower predicted_value
    #
    # Then score based on:
    #   - Frontier points get high scores
    #   - Within the frontier, reward points with better (lower) predicted_value
    #     and lower cost
    #   - Non-frontier points get scores based on how close they are to the frontier

    # Sort by predicted_value (ascending)
    sort_idx = np.argsort(predicted_value)
    sorted_pred = predicted_value[sort_idx]
    sorted_cost = cost[sort_idx]

    # Compute running minimum of cost
    running_min_cost = np.minimum.accumulate(sorted_cost)

    # A point is on the Pareto frontier if its cost is less than or equal
    # to the minimum cost of all points with strictly lower predicted_value
    # (i.e., the running min cost up to but not including this point)
    # For the first point, it's always on the frontier
    frontier_mask = np.zeros(n, dtype=bool)
    frontier_mask[sort_idx[0]] = True
    for i in range(1, n):
        idx = sort_idx[i]
        # The running min cost of all points before this one in sorted order
        min_cost_before = running_min_cost[i - 1]
        if sorted_cost[i] <= min_cost_before + 1e-12:
            frontier_mask[idx] = True

    # Score computation:
    # 1. Frontier points: score based on normalized (predicted_value, cost)
    # 2. Non-frontier points: lower score, based on distance to frontier

    # Normalize predicted_value to [0, 1] (lower is better, so invert)
    pred_min = np.min(predicted_value)
    pred_max = np.max(predicted_value)
    if pred_max > pred_min + 1e-12:
        pred_norm = (predicted_value - pred_min) / (pred_max - pred_min)
    else:
        pred_norm = np.zeros(n)

    # Normalize cost to [0, 1]
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min + 1e-12:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.ones(n)

    # For frontier points: high score based on combined normalized metrics
    # We want to reward low predicted_value and low cost
    frontier_score = 1.0 - 0.5 * pred_norm - 0.5 * cost_norm

    # For non-frontier points: penalize based on how far they are from the frontier
    # Use a simple penalty: score is negative, scaled by how dominated they are
    non_frontier_score = -1.0 - 0.5 * pred_norm - 0.5 * cost_norm

    # Combine
    score = np.where(frontier_mask, frontier_score, non_frontier_score)

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
