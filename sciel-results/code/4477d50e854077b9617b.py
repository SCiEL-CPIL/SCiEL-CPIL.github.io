"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Return one attractiveness score per item for the ant colony optimizer.
    The heuristic guides ants toward items that offer high prize-to-weight ratio
    across all constraints, with a smooth (log-sum-exp based) surrogate for the
    worst-case constraint tightness to avoid discontinuities.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    # Basic shape validation: expect (5, 100)
    # If shapes are wrong, return a safe default to avoid invalid results
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        return np.ones(100)

    # Ensure finiteness
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        return np.ones(100)

    # Sum of weights per item across constraints
    weight_sum = weights.sum(axis=0)  # shape (100,)

    # Sum of prizes per item across constraints (or use mean prize)
    # Use the sum of prizes as a proxy for total value contributed
    prize_sum = prizes.sum(axis=0)  # shape (100,)

    # Ratio of prize to weight, guarding against zero weight
    ratio = prize_sum / (weight_sum + 1e-9)

    # To create a more nuanced heuristic, we consider the "bottleneck" constraint.
    # For each item, the tightest constraint is the one where weight/capacity is highest.
    # Capacity is 25 for each constraint. So for constraint c, item i has load = weights[c,i]/25.
    # We want to prefer items that don't overly stress any single constraint.
    # Use a soft-max (log-sum-exp) of the normalized weights to smooth the max.
    # Normalized weight per constraint: weights / 25.0
    normalized_weights = weights / 25.0  # shape (5, 100)

    # Soft-max over constraints for each item: log(sum(exp(normalized_weights)))
    # This is a smooth upper bound on the max normalized weight.
    # Subtract max for numerical stability
    max_norm = normalized_weights.max(axis=0)  # shape (100,)
    shifted = normalized_weights - max_norm[:, None]  # shape (5, 100)
    soft_max = max_norm + np.log(np.sum(np.exp(shifted), axis=0))  # shape (100,)

    # The heuristic should reward high ratio but penalize items that stress constraints.
    # We combine: score = ratio * exp(-alpha * soft_max)
    # alpha controls the penalty strength. A moderate alpha balances exploration.
    alpha = 2.0
    penalty = np.exp(-alpha * soft_max)

    heuristic = ratio * penalty

    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Normalize to have mean 1 for consistency (optional, but helps with pheromone scaling)
    # The task says "larger is more attractive", absolute scale matters relative to pheromone.
    # Pheromone starts at 1, so heuristic ~1 is reasonable.
    mean_heur = heuristic.mean()
    if mean_heur > 0:
        heuristic = heuristic / mean_heur

    # Final safety: ensure positive
    heuristic = np.maximum(heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
