"""Open-shop dispatch rule: Time-dependent MWR/SPT hybrid heuristic."""
# EVOLVE-BLOCK-START
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _time_left():
    return _DEADLINE - time.time()

def dispatch(ready_operations, time):
    """
    Dispatch rule that balances Most Work Remaining (MWR) and Shortest Processing Time (SPT).
    - Early in the schedule (low 'time'), prioritize jobs with high remaining work (MWR).
    - Late in the schedule (high 'time'), prioritize operations with short processing times (SPT) to clear the queue.
    - Machine selection is always SPT (shortest duration on the chosen operation).
    """
    if not ready_operations:
        return 0

    # Safety check for time budget (though this heuristic is O(N*M) and very fast)
    if _time_left() < 0.01:
        return 0

    # Parameters for the time-dependent weighting
    # decay_rate controls how quickly we shift from MWR to SPT
    decay_rate = 0.15
    
    # Calculate alpha: weight for MWR
    # alpha = 1.0 at t=0, decreases as time increases
    alpha = 1.0 / (1.0 + time * decay_rate)
    
    # Weight for SPT is (1 - alpha)
    # We want to MAXIMIZE: alpha * remaining_work - (1 - alpha) * min_duration
    
    best_idx = -1
    best_score = -float('inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        
        # Find the machine with the minimum duration (SPT)
        min_d = float('inf')
        best_m = None
        for m, d in op["machine_options"].items():
            if d < min_d:
                min_d = d
                best_m = m
        
        # Calculate the composite score
        # We maximize remaining work (early) and minimize duration (late)
        # score = alpha * rw - (1 - alpha) * min_d
        
        # To avoid issues if rw and min_d have different scales, 
        # we assume they are comparable or that the heuristic holds.
        # In open shop, remaining_work is sum of durations, so it's usually larger than a single duration.
        # We might need to normalize.
        
        # Let's normalize by the max possible remaining work? 
        # We don't know the global max. 
        # Let's just use the raw values. If rw is 50 and min_d is 5, 
        # at alpha=0.5, score = 25 - 2.5 = 22.5.
        # If another op has rw=10, min_d=2, score = 5 - 1 = 4.
        # The MWR op wins.
        
        # If alpha=0.1, score = 5 - 4.5 = 0.5 vs 1 - 0.8 = 0.2. MWR still wins.
        # If alpha=0.01, score = 0.5 - 4.95 = -4.45 vs 0.1 - 1.98 = -1.88. SPT wins.
        
        score = alpha * rw - (1.0 - alpha) * min_d

        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
