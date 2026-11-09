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
    Uses adaptive weights based on schedule progress and tuned soft-min temperature.
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
    
    # Estimate total work to gauge schedule progress
    total_remaining = np.sum(remaining_works)
    # Heuristic: assume total work is roughly 5x current remaining (10 jobs, 5 machines)
    # This is a rough estimate for adaptive weighting
    estimated_total_work = total_remaining * 2.0 if total_remaining > 0 else 1.0
    progress = 1.0 - (total_remaining / estimated_total_work) if estimated_total_work > 0 else 0.0
    progress = np.clip(progress, 0.0, 1.0)
    
    # Soft-min surrogate for machine durations using log-sum-exp
    # Tuned temperature: 0.5 gives a sharper min, closer to hard min
    soft_min_durations = np.zeros(n, dtype=np.float64)
    temp = 0.5
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if machine_options:
            vals = np.array(list(machine_options.values()), dtype=np.float64)
            if len(vals) > 0:
                # Soft-min: -logsumexp(-x) ≈ min(x) but smooth
                shifted = -vals / temp
                max_shifted = np.max(shifted)
                if max_shifted > -1e10:
                    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
                    soft_min_durations[i] = -temp * log_sum_exp
                else:
                    soft_min_durations[i] = np.min(vals)
            else:
                soft_min_durations[i] = 0.0
        else:
            soft_min_durations[i] = 0.0
    
    # Critical path estimate: remaining_work + current_op_duration
    critical_path_scores = remaining_works + soft_min_durations
    
    # Normalize each feature
    max_mw = np.max(remaining_works) if np.max(remaining_works) > 0 else 1.0
    max_cp = np.max(critical_path_scores) if np.max(critical_path_scores) > 0 else 1.0
    max_mor = np.max(remaining_ops_counts) if np.max(remaining_ops_counts) > 0 else 1.0
    max_sm = np.max(soft_min_durations) if np.max(soft_min_durations) > 0 else 1.0
    
    normalized_mw = remaining_works / max_mw
    normalized_cp = critical_path_scores / max_cp
    normalized_mor = remaining_ops_counts / max_mor
    normalized_sm = soft_min_durations / max_sm
    
    # Adaptive weights: as progress increases, emphasize critical path more
    # Early: more weight on remaining work (longest job first)
    # Late: more weight on critical path (finish critical path first)
    w_cp = 0.3 + 0.3 * progress      # 0.3 -> 0.6
    w_mw = 0.4 - 0.2 * progress      # 0.4 -> 0.2
    w_sm = 0.2                        # fixed
    w_mor = 0.1                       # fixed
    
    # Normalize weights to sum to 1
    w_sum = w_cp + w_mw + w_sm + w_mor
    w_cp /= w_sum
    w_mw /= w_sum
    w_sm /= w_sum
    w_mor /= w_sum
    
    combined_scores = (
        w_cp * normalized_cp +
        w_mw * normalized_mw +
        w_sm * normalized_sm +
        w_mor * normalized_mor
    )
    
    # Return index of the operation with the highest combined score
    return int(np.argmax(combined_scores))
# EVOLVE-BLOCK-END
