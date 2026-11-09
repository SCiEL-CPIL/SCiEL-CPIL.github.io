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
    We use T=1.0 for a balance between sensitivity to min and averaging.
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
    Uses a load-balancing heuristic that estimates the impact of assigning each
    operation to its best machine on the overall makespan, utilizing a soft-min
    surrogate for processing times to capture machine option distributions.
    """
    if not ready_operations:
        return 0
    
    # Check if we're approaching the deadline - if so, use a fast fallback
    if _check_deadline():
        # Fast fallback: pick the operation with the most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    n_ops = len(ready_operations)
    if n_ops == 1:
        return 0

    # For each operation, compute features for load-balancing
    soft_min_durations = []
    remaining_works = []
    remaining_ops_counts = []
    flexibilities = []
    
    for op in ready_operations:
        mo = op["machine_options"]
        if mo:
            durations = list(mo.values())
            # Use soft-min to get a smooth estimate of the processing time
            # This is better than hard min as it penalizes options that are slightly longer
            soft_min_durations.append(float(_soft_min(durations, temperature=1.0)))
            flexibilities.append(len(mo))
        else:
            soft_min_durations.append(0.0)
            flexibilities.append(0)
        remaining_works.append(op["remaining_work"])
        remaining_ops_counts.append(op["remaining_ops"])
    
    # Critical path estimate: The operation contributes its current duration plus the rest of the job's work
    # Prioritizing high critical path helps minimize the makespan of the specific job
    critical_paths = [mw + sm for mw, sm in zip(remaining_works, soft_min_durations)]
    
    # Normalize features
    max_cp = max(critical_paths) if max(critical_paths) > 0 else 1
    max_sm = max(soft_min_durations) if max(soft_min_durations) > 0 else 1
    max_mw = max(remaining_works) if max(remaining_works) > 0 else 1
    max_mor = max(remaining_ops_counts) if max(remaining_ops_counts) > 0 else 1
    max_flex = max(flexibilities) if max(flexibilities) > 0 else 1
    
    normalized_cp = [x / max_cp for x in critical_paths]
    normalized_sm = [x / max_sm for x in soft_min_durations]
    normalized_mw = [x / max_mw for x in remaining_works]
    normalized_mor = [x / max_mor for x in remaining_ops_counts]
    # Invert flexibility: less flexible = higher score (more constrained ops first)
    normalized_flex_inv = [1.0 - (x / max_flex) for x in flexibilities]
    
    # Combine with weights for load balancing and critical path awareness:
    # Critical Path (0.40): Most important for makespan reduction
    # Soft-min Duration (0.30): Longest processing time first aspect
    # Remaining Work (0.20): Secondary indicator of job size
    # Constrained (0.05): Tie-breaker for flexibility
    # MOR (0.05): Tie-breaker for sequence length
    
    combined_scores = [
        0.40 * cp + 0.30 * sm + 0.20 * mw + 0.05 * flex_inv + 0.05 * mor
        for cp, sm, mw, flex_inv, mor in zip(
            normalized_cp, normalized_sm, normalized_mw, 
            normalized_flex_inv, normalized_mor
        )
    ]
    
    return int(np.argmax(combined_scores))
# EVOLVE-BLOCK-END
