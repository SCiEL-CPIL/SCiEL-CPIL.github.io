"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import math

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 140.0  # Safety margin before 150s kill

# Global state to track progress for dynamic weighting
_initial_total_work = None
_max_total_work_seen = 0.0

def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next (and optionally which machine).
    
    Strategy:
    1. Dynamic Blend of LRPT (Longest Remaining Processing Time) and SPT (Shortest Processing Time).
       - Early in the schedule: Prioritize LRPT to keep the critical path moving.
       - Late in the schedule: Prioritize SPT to clear the tail and reduce idle time.
    2. Machine Selection: Choose the machine with the minimum duration for the selected operation.
    
    Returns: (index, machine) tuple.
    """
    global _initial_total_work, _max_total_work_seen
    
    # Deadline guard
    if _time.time() > _DEADLINE:
        if ready_operations:
            # Fallback: Pick the one with the most remaining work
            best_idx = 0
            max_rw = -1
            for i, op in enumerate(ready_operations):
                rw = op.get("remaining_work", 0)
                if rw > max_rw:
                    max_rw = rw
                    best_idx = i
            return best_idx
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    
    # Calculate total remaining work in the ready queue to estimate progress
    current_total_work = 0.0
    for op in ready_operations:
        current_total_work += op.get("remaining_work", 0)
        
    # Initialize global trackers
    if _initial_total_work is None:
        _initial_total_work = current_total_work
        _max_total_work_seen = current_total_work
    else:
        _max_total_work_seen = max(_max_total_work_seen, current_total_work)

    # Estimate progress: 0 at start, 1 at end
    # If initial work is 0, assume late game
    if _initial_total_work > 0:
        # We don't know the exact end, but if current work is much lower than initial, we are late.
        # However, work is added to the queue as jobs complete. 
        # A better proxy for "late game" is the average remaining operations count.
        avg_remaining_ops = sum(op.get("remaining_ops", 0) for op in ready_operations) / n
        # Assume a typical job has ~5-10 ops. If avg is < 2, we are in the tail.
        # Let's use a smooth transition.
        # progress = 1.0 - (avg_remaining_ops / 5.0) # Crude
        # Better: Use the ratio of current min remaining work to max seen?
        
        # Let's stick to a time-based or work-based heuristic.
        # If we assume the total work processed is roughly linear with time, and we don't know the total horizon,
        # we can use the count of operations.
        # Let's use a dynamic weight based on the average remaining operations.
        # High avg ops -> Early -> LRPT (alpha=1.0)
        # Low avg ops -> Late -> SPT (alpha=0.5)
        
        # Map avg_remaining_ops from [0, 10] to [0.5, 1.0]
        # If avg is 0, alpha = 0.5. If avg >= 10, alpha = 1.0.
        alpha = 0.5 + 0.5 * min(1.0, max(0.0, avg_remaining_ops / 5.0))
        
    else:
        alpha = 0.5

    best_machine_indices = []
    min_durations = []
    remaining_works = []
    
    for i, op in enumerate(ready_operations):
        mw = op.get("remaining_work", 0)
        machine_options = op.get("machine_options", {})
        
        if machine_options:
            durations = list(machine_options.values())
            machines = list(machine_options.keys())
            
            min_dur = min(durations)
            best_machine = machines[durations.index(min_dur)]
            
            min_durations.append(min_dur)
            best_machine_indices.append(best_machine)
        else:
            min_durations.append(0)
            best_machine_indices.append(None)
            
        remaining_works.append(mw)

    # Normalize for scoring
    max_rw = max(remaining_works) if remaining_works else 1
    max_min_dur = max(min_durations) if min_durations else 1
    min_min_dur = min(min_durations) if min_durations else 0

    scores = []
    for i in range(n):
        # LRPT Component: Higher remaining work is better
        lrpt_score = remaining_works[i] / max_rw if max_rw > 0 else 0
        
        # SPT Component: Lower duration is better
        if max_min_dur > min_min_dur:
            # Normalize duration to [0, 1] where 0 is best (shortest)
            # raw_norm = (dur - min) / (max - min)
            # spt_score = 1 - raw_norm
            raw_norm = (min_durations[i] - min_min_dur) / (max_min_dur - min_min_dur)
            spt_score = 1.0 - raw_norm
        else:
            spt_score = 1.0
        
        # Composite Score
        # alpha controls the blend. 
        # Early: alpha ~ 1.0 -> Score ~ LRPT
        # Late: alpha ~ 0.5 -> Score ~ 0.5 LRPT + 0.5 SPT
        score = alpha * lrpt_score + (1.0 - alpha) * spt_score
        scores.append(score)

    # Find the index with maximum score
    best_idx = 0
    best_score = scores[0]
    for i in range(1, n):
        if scores[i] > best_score:
            best_score = scores[i]
            best_idx = i

    # Return (index, machine) to allow explicit machine selection
    if best_machine_indices[best_idx] is not None:
        return (best_idx, best_machine_indices[best_idx])
    else:
        return best_idx
# EVOLVE-BLOCK-END
