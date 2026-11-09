"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_DEADLINE_START = _time.time()
_MAX_WALL = 140.0  # Safety margin before the 150s kill


def _check_deadline(fallback):
    """If we're getting close to the wall-clock limit, return fallback immediately."""
    if _time.time() - _DEADLINE_START > _MAX_WALL:
        return fallback
    return None


def dispatch(ready_operations, time):
    # Check deadline first; if we're over, pick the first available (index 0)
    fallback = 0
    early = _check_deadline(fallback)
    if early is not None:
        return early

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Strategy: Refined LRPT with LPT Tie-Breaker
    # 1. Prioritize jobs with the most remaining work (LRPT).
    # 2. Tie-breaker: Prioritize operations with the longest maximum processing time
    #    among available machines (LPT). This ensures long, potentially bottlenecking
    #    operations are started early.
    # 3. Final Tie-breaker: Shortest minimum duration (SPT) to clear short tasks if all else equal.

    best_idx = 0
    best_remaining_work = -1.0
    best_max_duration = -1.0
    best_min_duration = float('inf')

    for i, op in enumerate(ready_operations):
        rw = op["remaining_work"]
        
        # Find min and max duration among available machines for this operation
        options = op["machine_options"]
        if not options:
            min_dur = 0.0
            max_dur = 0.0
        else:
            vals = list(options.values())
            min_dur = min(vals)
            max_dur = max(vals)

        # Compare with current best
        # 1. Higher remaining work is better
        if rw > best_remaining_work:
            best_idx = i
            best_remaining_work = rw
            best_max_duration = max_dur
            best_min_duration = min_dur
        elif abs(rw - best_remaining_work) < 1e-9:
            # 2. If remaining work is effectively the same, pick the one with LONGER max duration
            if max_dur > best_max_duration:
                best_idx = i
                best_max_duration = max_dur
                best_min_duration = min_dur
            elif abs(max_dur - best_max_duration) < 1e-9:
                # 3. If max duration is also same, pick the one with SHORTER min duration
                if min_dur < best_min_duration:
                    best_idx = i
                    best_min_duration = min_dur

    return best_idx
# EVOLVE-BLOCK-END
