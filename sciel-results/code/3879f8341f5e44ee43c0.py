"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Uses a 'bottleneck efficiency' heuristic: prize divided by the maximum weight
    across all constraints. This models the fact that an item's inclusion is limited
    by its most restrictive constraint, rather than the sum of all constraints.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit

    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Check for immediate deadline breach (unlikely for O(N) but good practice)
    if time.time() > deadline:
        return np.ones(prizes.shape[1])

    # Calculate the bottleneck weight for each item (max weight across constraints)
    # This identifies the constraint that will be hit first if we add this item.
    max_weights = np.max(weights, axis=0)
    
    # Calculate efficiency: prize / bottleneck weight
    # Add epsilon to avoid division by zero
    epsilon = 1e-9
    heuristic = prizes / (max_weights + epsilon)
    
    # Ensure all values are positive and finite
    # The task states: "Heuristic entries that are not positive become 1e-9"
    # We handle non-finite values by replacing them with a small positive number
    # and ensuring positivity.
    
    # Replace NaN or Inf with 0, then set non-positive to 1e-9
    heuristic = np.where(np.isfinite(heuristic), heuristic, 0.0)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
