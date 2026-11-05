"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item to add to the knapsack using a 1-step lookahead
    heuristic with greedy completion.
    
    For each feasible candidate item, it simulates the rest of the greedy
    (by density) filling process to estimate the total value of the solution
    if that item is chosen. It picks the candidate that yields the highest
    estimated total value.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    # Ensure arrays are numpy arrays for potential vectorization or just consistent typing
    weights = np.array(remaining_weights)
    values = np.array(remaining_values)
    
    # Identify feasible items (weight <= capacity)
    # We only consider these as candidates for the first pick
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1

    # Calculate densities for the greedy completion simulation
    # Avoid division by zero, though weights should be positive
    densities = values / weights
    
    # Sort indices by density in descending order
    # This order is used for the "greedy completion" part of the simulation
    sorted_indices = np.argsort(-densities)

    best_val = -1
    best_idx = -1

    # Iterate through all feasible candidates
    # We use the original indices to map back to the input arrays
    feasible_indices = np.where(feasible_mask)[0]
    
    for k in feasible_indices:
        # Simulate picking item k
        cap_rem = remaining_capacity - weights[k]
        val_acc = values[k]
        
        # Greedily fill the remaining capacity with items sorted by density
        # We skip item k itself
        for idx in sorted_indices:
            if idx == k:
                continue
            if weights[idx] <= cap_rem:
                cap_rem -= weights[idx]
                val_acc += values[idx]
                if cap_rem <= 0:
                    break
        
        if val_acc > best_val:
            best_val = val_acc
            best_idx = k

    return int(best_idx)
# EVOLVE-BLOCK-END
