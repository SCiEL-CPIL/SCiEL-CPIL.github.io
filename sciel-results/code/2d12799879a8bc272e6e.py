"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

# Module-level state for deadline guard
_start_time = _time.monotonic()
_DEADLINE = 150.0  # seconds
_MARGIN = 5.0      # safety margin in seconds


def dispatch(ready_operations, time):
    """
    Dispatch rule for flexible job-shop scheduling.
    
    Returns: index of the ready operation to start, or (index, machine) tuple.
    
    Strategy:
    - Use a composite heuristic that considers:
      1. Minimum processing time (MPT) - shorter operations first to reduce idle
      2. Remaining work - prioritize jobs with more work left (LWR)
      3. Machine load balancing - choose the least loaded eligible machine
    - Select the operation with the best composite score, and pick the
      shortest-duration machine for that operation to minimize makespan.
    """
    # Deadline guard: if we're close to the wall-clock limit, use a fast fallback
    if _time.monotonic() - _start_time > _DEADLINE - _MARGIN:
        # Fast fallback: pick the operation with minimum processing time
        best_idx = 0
        best_dur = float('inf')
        for i, op in enumerate(ready_operations):
            opts = op["machine_options"]
            min_dur = min(opts.values())
            if min_dur < best_dur:
                best_dur = min_dur
                best_idx = i
        return best_idx

    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        return 0

    # Compute scores for each ready operation
    best_idx = 0
    best_score = float('-inf')
    best_machine = None

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        # Find the machine with the shortest duration for this operation
        min_dur = float('inf')
        chosen_machine = None
        for m, d in opts.items():
            if d < min_dur:
                min_dur = d
                chosen_machine = m

        remaining_work = op["remaining_work"]
        remaining_ops = op["remaining_ops"]

        # Composite score:
        # - Primary: minimize min processing time (MPT)
        # - Secondary: maximize remaining work (LWR) to avoid stragglers
        # - Tertiary: prefer operations with fewer remaining ops (closer to completion)
        score = -min_dur + 0.1 * remaining_work - 0.01 * remaining_ops

        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = chosen_machine

    # Return (index, machine) to also guide machine selection
    if best_machine is not None:
        return (best_idx, best_machine)
    return best_idx
# EVOLVE-BLOCK-END
