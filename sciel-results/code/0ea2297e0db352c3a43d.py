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

    n_items = prize_vector.shape[0]
    n_constraints = weights.shape[0]
    capacity = 25.0

    # Calculate sum and max weights per item
    weight_sums = np.sum(weights, axis=0)
    weight_maxs = np.max(weights, axis=0)

    # Bottleneck slack: for each item, compute the minimum remaining capacity ratio
    # across all constraints if that item were added first.
    # slack_c = (capacity - w_ic) / w_ic for each constraint c
    # bottleneck_slack = min over c of slack_c
    # This directly measures how "easy" it is to fit the item in the tightest constraint.
    
    # Avoid division by zero
    safe_weights = np.maximum(weights, 1e-9)
    slack_ratios = (capacity - weights) / safe_weights
    bottleneck_slack = np.min(slack_ratios, axis=0)
    
    # Ensure bottleneck_slack is non-negative (items that don't fit get 0 or negative)
    bottleneck_slack = np.maximum(bottleneck_slack, 0.0)
    
    # Combine signals:
    # 1. Prize density: prize_vector / (weight_sums + eps)
    # 2. Bottleneck slack: how easily the item fits in the tightest constraint
    # 3. Total prize: higher absolute prize is better
    
    # Primary signal: prize * bottleneck_slack
    # This rewards items that have high prize AND fit easily in the tightest constraint
    primary = prize_vector * bottleneck_slack
    
    # Secondary signal: prize density
    density = prize_vector / (weight_sums + 1e-9)
    
    # Combine: use a weighted combination
    # Normalize both to [0, 1] range for fair combination
    primary_max = np.max(primary) if np.max(primary) > 0 else 1.0
    primary_norm = primary / primary_max
    
    density_max = np.max(density) if np.max(density) > 0 else 1.0
    density_norm = density / density_max
    
    # Weight the combination: emphasize bottleneck slack more as it captures feasibility
    heuristic = 0.6 * primary_norm + 0.4 * density_norm
    
    # Apply a moderate power to sharpen the distribution
    heuristic = heuristic ** 1.2
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Check time
    if time.time() > deadline:
        return heuristic

    return heuristic

# EVOLVE-BLOCK-END
