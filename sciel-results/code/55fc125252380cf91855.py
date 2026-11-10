"""Combined program: bottleneck-slack feasibility scoring with composite density."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    # Deadline guard for the whole evaluation budget.
    start_time = time.time()
    deadline = 300.0 - 5.0

    # Convert inputs to float arrays.
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Handle edge cases
    if prizes.size == 0 or weights.size == 0:
        return np.array([], dtype=float)

    # Reduce prizes to a 1-D vector of per-item total prize.
    if prizes.ndim == 2:
        prize_vector = np.sum(prizes, axis=0)
    else:
        prize_vector = prizes.flatten()

    # Per-item weight statistics across the 5 constraints.
    w_sum = np.sum(weights, axis=0)
    w_max = np.max(weights, axis=0)

    # Composite effective weight: captures both binding constraint (max)
    # and total resource consumption (sum).
    # Using 0.8 coefficient for max weight to strongly penalize bottleneck-heavy items.
    effective_weight = w_max + 0.8 * w_sum + 1e-12

    # Base density: prize / effective_weight
    density = prize_vector / effective_weight

    # Bottleneck slack score: for each item, compute min_c (25 - w_ic) / w_ic
    # This measures how easily the item fits into the tightest constraint.
    capacity = 25.0
    # To avoid division by zero, add epsilon
    w_safe = np.maximum(weights, 1e-12)
    slack_matrix = (capacity - weights) / w_safe  # shape (5, 100)
    bottleneck_slack = np.min(slack_matrix, axis=0)  # shape (100,)
    
    # Clip negative slack to a small positive value to keep heuristic positive
    # Items that don't fit alone will have very low slack, which is appropriate
    bottleneck_slack = np.maximum(bottleneck_slack, 1e-9)

    # Combine: prize * bottleneck_slack * density^power
    # The bottleneck_slack directly prioritizes items that fit easily into tight constraints
    # The density term captures value efficiency
    # Use a moderate power to sharpen but not over-concentrate
    power = 1.5
    heuristic = (prize_vector * bottleneck_slack) * (density ** power)

    # Guarantee positivity and finiteness.
    heuristic = np.maximum(heuristic, 1e-9)
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)

    # If we are running out of wall-clock, return immediately.
    if time.time() > deadline:
        return heuristic

    return heuristic

# EVOLVE-BLOCK-END
