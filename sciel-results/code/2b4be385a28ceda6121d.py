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


def _soft_min(values, temperature=1.0):
    """
    Compute a soft-min of values using log-sum-exp trick.
    soft_min(x) = -T * log(sum(exp(-x_i / T)))
    As T -> 0, this approaches the true minimum.
    As T -> inf, this approaches the mean.
    """
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    # Shift for numerical stability
    min_val = np.min(arr)
    exp_terms = np.exp(-(arr - min_val) / temperature)
    log_sum = np.log(np.sum(exp_terms))
    return min_val + temperature * log_sum


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    Uses a hybrid heuristic with soft-min surrogate for machine duration distribution.
    """
    if not ready_operations:
        return 0
    
    # Check if we're approaching the deadline - if so, use a fast fallback
    if _check_deadline():
        # Fast fallback: pick the operation with the most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Calculate soft-min durations from machine_options
    # This captures the full distribution of machine options, not just the minimum
    soft_min_durations = []
    for op in ready_operations:
        machine_options = op["machine_options"]
        if machine_options:
            durations = list(machine_options.values())
            # Use temperature=1.0 for a smooth approximation of the minimum
            soft_min_dur = _soft_min(durations, temperature=1.0)
        else:
            soft_min_dur = 0
        soft_min_durations.append(soft_min_dur)
    
    # Extract features
    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops_counts = [op["remaining_ops"] for op in ready_operations]
    
    # Heuristic 1: Most work remaining (MW)
    mw_scores = remaining_works
    
    # Heuristic 2: Longest processing time first (LPT) - use soft-min duration
    lpt_scores = soft_min_durations
    
    # Heuristic 3: Most operations remaining
    mor_scores = remaining_ops_counts
    
    # Combine heuristics with weights
    # Weighted combination: prioritize processing time (captures machine load), then work, then ops
    # Normalize scores to comparable scales
    max_mw = max(mw_scores) if max(mw_scores) > 0 else 1
    max_lpt = max(lpt_scores) if max(lpt_scores) > 0 else 1
    max_mor = max(mor_scores) if max(mor_scores) > 0 else 1
    
    normalized_mw = [x / max_mw for x in mw_scores]
    normalized_lpt = [x / max_lpt for x in lpt_scores]
    normalized_mor = [x / max_mor for x in mor_scores]
    
    # Combine with weights: LPT (0.5), MW (0.3), MOR (0.2)
    # Prioritize processing time as it directly impacts makespan
    combined_scores = [
        0.5 * lpt + 0.3 * mw + 0.2 * mor
        for mw, lpt, mor in zip(normalized_mw, normalized_lpt, normalized_mor)
    ]
    
    # Return index of the operation with the highest combined score
    return int(np.argmax(combined_scores))
# EVOLVE-BLOCK-END
