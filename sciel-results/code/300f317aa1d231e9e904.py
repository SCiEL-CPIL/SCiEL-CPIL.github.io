"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Track start time for deadline guard
_START_TIME = time.time()
DEADLINE_MARGIN = 5.0  # seconds before the 150s limit


def _check_deadline():
    """Return True if we're getting close to the deadline."""
    return (time.time() - _START_TIME) > (150.0 - DEADLINE_MARGIN)


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    
    Strategy: Combine multiple heuristics:
    1. Critical path / remaining work (job urgency)
    2. Minimum processing time on available machines (shortest job first)
    3. Machine load balancing (prefer less-loaded machines)
    4. Remaining operations count (jobs with fewer remaining ops are more urgent)
    
    We normalize each feature to [0,1] across the ready operations and use a weighted sum.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Check deadline - if close, use fast fallback
    if _check_deadline():
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))
    
    # Extract features
    remaining_work = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # For each operation, find the minimum duration across machine options
    min_durations = np.array([
        min(op["machine_options"].values()) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=float)
    
    # Number of machine options (flexibility)
    n_machines = np.array([
        len(op["machine_options"]) if op["machine_options"] else 0
        for op in ready_operations
    ], dtype=float)
    
    # Normalize features to [0, 1]
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)
    
    norm_work = normalize(remaining_work)
    norm_ops = normalize(remaining_ops)
    norm_min_dur = normalize(min_durations)
    norm_nmach = normalize(n_machines)
    
    # Weights:
    # - High remaining work -> high priority (critical path)
    # - Few remaining ops -> high priority (close to completion)
    # - Short min duration -> high priority (quick completion frees machine)
    # - Few machine options -> slightly higher priority (less flexibility, schedule sooner)
    
    # Score: we want to maximize this
    # Invert min_dur and nmach so higher = more urgent
    score = (
        0.45 * norm_work +       # Most remaining work first (critical path)
        0.25 * (1.0 - norm_ops) + # Fewer remaining ops first
        0.20 * (1.0 - norm_min_dur) +  # Shorter processing time first
        0.10 * (1.0 - norm_nmach)      # Less flexibility first
    )
    
    # Tie-breaking: prefer shorter min_duration, then higher remaining work
    best_idx = int(np.argmax(score))
    
    # Check for near-ties and break deterministically
    max_score = score[best_idx]
    tie_indices = np.where(score > max_score - 1e-6)[0]
    if len(tie_indices) > 1:
        # Among ties, pick the one with shortest min_duration
        tie_scores = min_durations[tie_indices]
        best_tie = np.argmin(tie_scores)
        best_idx = int(tie_indices[best_tie])
    
    return best_idx
# EVOLVE-BLOCK-END
