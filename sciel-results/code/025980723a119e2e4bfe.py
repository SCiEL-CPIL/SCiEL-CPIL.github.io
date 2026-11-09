"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# Global deadline guard to ensure we finish well within the 120s evaluation limit
_START_TIME = _time.time()
_HARD_LIMIT = 110.0  # seconds
_SOFT_MARGIN = 5.0   # seconds buffer before hard limit


def _check_deadline():
    """Return True if we are approaching the hard limit."""
    return (_time.time() - _START_TIME) > (_HARD_LIMIT - _SOFT_MARGIN)


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    Uses a hybrid heuristic with soft-min surrogate and critical-path awareness.
    """
    if not ready_operations:
        return 0
    
    # Check if we're approaching the deadline - if so, use a fast fallback
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Extract features
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=np.float64)
    remaining_ops_counts = np.array([op["remaining_ops"] for op in ready_operations], dtype=np.float64)
    
    # Soft-min surrogate for machine durations using log-sum-exp
    # This captures the full distribution, not just the minimum
    # For each op, compute a soft-min of its machine_options values
    soft_min_durations = np.zeros(n, dtype=np.float64)
    min_durations = np.zeros(n, dtype=np.float64)
    max_durations = np.zeros(n, dtype=np.float64)
    
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if machine_options:
            vals = np.array(list(machine_options.values()), dtype=np.float64)
            if len(vals) > 0:
                min_durations[i] = np.min(vals)
                max_durations[i] = np.max(vals)
                # Soft-min: -logsumexp(-x) ≈ min(x) but smooth
                # Use temperature parameter for control
                # Lower temp emphasizes the minimum more strongly
                temp = 0.5
                shifted = -vals / temp
                max_shifted = np.max(shifted)
                if max_shifted > -1e10:
                    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
                    soft_min_durations[i] = -temp * log_sum_exp
                else:
                    soft_min_durations[i] = np.min(vals)
            else:
                soft_min_durations[i] = 0.0
                min_durations[i] = 0.0
                max_durations[i] = 0.0
        else:
            soft_min_durations[i] = 0.0
            min_durations[i] = 0.0
            max_durations[i] = 0.0
    
    # Bipartite graph approach: model edges between ready operations and machines.
    # Edge weight = remaining_work / processing_time on that machine.
    # This represents the reduction in critical path per unit of machine time.
    # Select the operation with the highest maximum edge weight (best machine assignment).
    
    # For each operation, compute the maximum edge weight across all machine options
    # Edge weight for op i on machine m = remaining_work[i] / duration[i][m]
    # Higher weight means more work cleared per unit time on that machine
    
    max_edge_weights = np.zeros(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if machine_options and remaining_works[i] > 0:
            vals = np.array(list(machine_options.values()), dtype=np.float64)
            # Avoid division by zero
            safe_vals = np.where(vals > 0, vals, 1e-10)
            edge_weights = remaining_works[i] / safe_vals
            max_edge_weights[i] = np.max(edge_weights)
        else:
            max_edge_weights[i] = 0.0
    
    # Also compute a secondary score: remaining_work / soft_min_duration
    # This captures the full distribution of machine options
    secondary_scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if soft_min_durations[i] > 0 and remaining_works[i] > 0:
            secondary_scores[i] = remaining_works[i] / soft_min_durations[i]
        else:
            secondary_scores[i] = 0.0
    
    # Normalize both scores
    max_mew = np.max(max_edge_weights) if np.max(max_edge_weights) > 0 else 1.0
    max_sec = np.max(secondary_scores) if np.max(secondary_scores) > 0 else 1.0
    
    normalized_mew = max_edge_weights / max_mew
    normalized_sec = secondary_scores / max_sec
    
    # Combine: primary is the bipartite max edge weight (0.60),
    # secondary is the soft-min based ratio (0.25),
    # remaining work as tiebreaker (0.15)
    max_mw = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    normalized_mw = remaining_works / max_mw
    
    combined_scores = (
        0.60 * normalized_mew +
        0.25 * normalized_sec +
        0.15 * normalized_mw
    )
    
    # Return index of the operation with the highest combined score
    return int(np.argmax(combined_scores))
# EVOLVE-BLOCK-END
