"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

# We use a heuristic that primarily favors Shortest Processing Time (SPT)
# with a Most Work Remaining (MWR) tie-breaker/adjustment.
# 
# Rationale:
# - SPT (picking the operation with the shortest min duration) tends to clear
#   the schedule quickly and reduce machine idle time, which is generally strong
#   for minimizing makespan in job-shop scheduling.
# - MWR (favoring jobs with more remaining work) helps avoid starvation and
#   ensures that long jobs make progress, but pure MWR can block machines with
#   long operations unnecessarily.
#
# Strategy:
# 1. Compute the minimum processing time for each ready operation across its machine options.
# 2. Use SPT as the primary criterion: operations with shorter min_duration get higher priority.
# 3. Add a small MWR-based adjustment to break ties and slightly favor jobs with more work remaining.
# 4. Use a weighted combination: score = w_spt * (1 / (min_duration + eps)) + w_mrw * remaining_work_normalized
#
# We normalize remaining_work within the current set of ready operations to keep the
# MWR component on a comparable scale.

def dispatch(ready_operations, time):
    if not ready_operations:
        return 0
    
    n = len(ready_operations)
    if n == 1:
        return 0
    
    # Compute min_duration and remaining_work for each operation
    min_durations = np.zeros(n)
    remaining_works = np.zeros(n)
    
    for i, op in enumerate(ready_operations):
        machine_options = op.get("machine_options", {})
        if machine_options:
            min_durations[i] = min(machine_options.values())
        else:
            min_durations[i] = 1.0
        
        remaining_works[i] = op.get("remaining_work", 0)
    
    # Avoid division by zero
    min_durations_safe = min_durations + 1e-6
    
    # SPT component: higher is better (shorter duration -> higher score)
    # Use log-sum-exp style soft-min approximation for smoother behavior
    # Instead of 1/min_duration, use a softer version that's less extreme
    spt_scores = 1.0 / min_durations_safe
    
    # Normalize remaining_work to [0, 1] within the current set
    rw_min = np.min(remaining_works)
    rw_max = np.max(remaining_works)
    if rw_max > rw_min:
        mrw_scores = (remaining_works - rw_min) / (rw_max - rw_min)
    else:
        mrw_scores = np.zeros(n)
    
    # Combine: SPT is primary, MWR is a secondary adjustment
    # Weights: 0.85 for SPT, 0.15 for MWR - slightly increase MWR weight
    w_spt = 0.85
    w_mrw = 0.15
    
    scores = w_spt * spt_scores + w_mrw * mrw_scores
    
    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))
    
    return best_idx
# EVOLVE-BLOCK-END
