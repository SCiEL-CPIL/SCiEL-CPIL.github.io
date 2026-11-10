"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    start_time = time.time()
    deadline = 300.0 - 5.0  # Safety margin

    # Convert to numpy arrays for consistent behavior
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Handle potential 2D prizes by reducing to 1D if necessary
    if prizes.ndim == 2:
        prize_vector = np.sum(prizes, axis=0)
    else:
        prize_vector = prizes.flatten()

    # Calculate weight metrics per item
    weight_sums = np.sum(weights, axis=0)
    weight_maxs = np.max(weights, axis=0)

    # Composite effective weight: captures both binding constraint (max)
    # and total resource consumption (sum). This was the key insight from
    # inspiration 1 which scored 1.0549 vs current's 1.0194.
    # effective_weight = max_weight + 0.5 * sum_weights
    effective_weight = weight_maxs + 0.5 * weight_sums

    # Base heuristic: prize / effective_weight
    heuristic = prize_vector / (effective_weight + 1e-9)

    # Apply power amplification to sharpen ACO focus on high-value-density items.
    # Power of 2.0 balances between too-sharp (2.5) and too-soft (1.1).
    # This makes the probability distribution steeper, helping ants converge
    # faster on the best candidates.
    heuristic = heuristic ** 2.0

    # Ensure positive and finite values
    heuristic = np.maximum(heuristic, 1e-9)
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)

    # Check time
    if time.time() > deadline:
        return heuristic

    return heuristic

# EVOLVE-BLOCK-END
