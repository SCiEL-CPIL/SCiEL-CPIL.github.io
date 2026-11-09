"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 145.0  # safety margin before 150s kill
_START_TIME = None


def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _time.monotonic()
    return _START_TIME


def _is_deadline_exceeded():
    return (_time.monotonic() - _ensure_start()) > _DEADLINE


def dispatch(ready_operations, time):
    """
    Dynamic bottleneck-aware dispatch rule for job-shop scheduling.
    
    Strategy: Select the operation whose assignment to its least-loaded machine
    maximizes the minimum slack time across all currently ready operations.
    This actively prevents machine starvation and reduces idle gaps.
    
    Returns index of the ready operation to start.
    """
    _ensure_start()
    
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0
    
    # Extract features for each operation
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)
    
    # For each operation, find the minimum duration machine option
    min_durations = np.array([min(op["machine_options"].values()) for op in ready_operations], dtype=float)
    max_durations = np.array([max(op["machine_options"].values()) for op in ready_operations], dtype=float)
    
    # Number of machine options (flexibility)
    n_options = np.array([len(op["machine_options"]) for op in ready_operations], dtype=float)
    
    # Compute slack estimates: how much "buffer" each operation has
    # Slack is proportional to remaining work relative to its processing time
    # Higher slack = more flexible timing
    slack_estimates = remaining_works / np.maximum(min_durations, 1e-9)
    
    # For each candidate operation i, simulate assigning it to its least-loaded machine
    # (approximated by min_duration machine) and compute the resulting min slack
    # across all other ready operations
    
    best_score = -np.inf
    best_idx = 0
    
    for i in range(n):
        if _is_deadline_exceeded():
            # Fallback to simple heuristic
            break
        
        # If we assign operation i, its machine will be busy for min_durations[i]
        # This affects the slack of other operations that might need the same machine
        
        # Estimate the impact: operations with fewer options are more affected
        # by machine contention
        
        # Compute a "post-assignment" slack for each operation
        # Operation i is now scheduled, so it's no longer in contention
        # Other operations' slack is reduced if they share machine constraints
        
        # Simple model: the bottleneck is the operation with the least slack
        # after considering that machine i is now occupied
        
        # For each other operation j, estimate how much its available time is reduced
        # if it needed the same machine as i
        post_slacks = np.zeros(n)
        for j in range(n):
            if j == i:
                continue
            
            # Base slack for operation j
            base_slack = slack_estimates[j]
            
            # If j has fewer machine options, it's more constrained
            # Estimate contention penalty: operations with fewer options
            # are more likely to be affected by machine i being busy
            contention_penalty = 0.0
            if n_options[j] <= 2:
                # Very constrained: significant penalty if sharing machine
                contention_penalty = min_durations[i] * 0.5
            elif n_options[j] <= 3:
                # Somewhat constrained
                contention_penalty = min_durations[i] * 0.25
            
            post_slacks[j] = base_slack - contention_penalty
        
        # The minimum slack among remaining operations determines the bottleneck
        # We want to maximize this minimum slack
        min_slack_after = np.min(post_slacks) if n > 1 else 0
        
        # Also consider the urgency of operation i itself
        # Operations with more remaining work should be prioritized
        urgency_bonus = remaining_works[i] / max(remaining_works.max(), 1e-9)
        
        # Combine: primary objective is max min-slack, secondary is urgency
        score = min_slack_after * 0.6 + urgency_bonus * 0.4
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    if _is_deadline_exceeded():
        # Fallback: use simple WSPT-like heuristic
        wspt = remaining_works / np.maximum(min_durations, 1e-9)
        best_idx = int(np.argmax(wspt))
    
    return int(best_idx)
# EVOLVE-BLOCK-END
