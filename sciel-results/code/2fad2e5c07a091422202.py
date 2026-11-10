"""Improved heuristic for ant colony offline bin packing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)

    # Pairwise sum of item sizes
    pair = d[:, None] + d[None, :]

    # A pair is "feasible" if their sum <= capacity
    feasible = pair <= cap

    # Remaining space after placing both items
    remaining = cap - pair

    # Use a smooth surrogate for the gap penalty:
    # Instead of 1/(remaining + eps), use exp(-remaining / scale) which is smooth
    # and provides a softer reward for near-full bins
    scale = cap / 4.0  # Tunable scale parameter
    gap_penalty = np.exp(-remaining / scale)
    
    # For infeasible pairs, set to a tiny value
    scores = np.where(feasible, gap_penalty, 1e-9)
    
    # Add a bonus for pairs where items are similar in size (helps with fragmentation)
    size_diff = np.abs(d[:, None] - d[None, :])
    similarity_bonus = 1.0 / (1.0 + size_diff / (cap / 2.0))
    
    # Combine: primary gap penalty with a similarity component
    # Increase similarity weight from 0.1 to 0.15 to better guide against fragmentation
    scores = scores * (1.0 + 0.15 * similarity_bonus)
    
    # Ensure all values are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)

    return scores
# EVOLVE-BLOCK-END
