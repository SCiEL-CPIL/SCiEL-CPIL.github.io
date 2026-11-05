"""Job-Shop Scheduling: Improved Multi-Criteria Dispatch Rule."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Global deadline guard: evaluator kills at 150s, use 140s as hard limit
_DEADLINE = 140.0
_START_TIME = _time.monotonic()

# Weights for the multi-criteria scoring function.
# We balance:
# 1. SPT (Shortest Processing Time): 1 / min_duration. Favors short ops to reduce queueing.
# 2. MWR (Most Work Remaining): remaining_work. Favors jobs with more work to keep pipeline full.
# 3. Finish Soon: 1 / remaining_ops. Favors jobs close to completion to clear the board.

_W_SPT = 1.0
_W_MWR = 0.1
_W_FINISH = 0.5


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Args:
        ready_operations: list of dicts, each with:
            - job: job id
            - op: operation index
            - machine_options: {machine: duration}
            - remaining_ops: number of ops remaining in this job
            - remaining_work: total remaining processing time for this job
        time: current simulation time
    
    Returns:
        index of the ready operation to start
    """
    # Deadline guard: if we're close to the limit, just return first op
    elapsed = _time.monotonic() - _START_TIME
    if elapsed > _DEADLINE - 5.0:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)

    if n == 1:
        return 0

    # Compute scores
    scores = np.zeros(n)

    for i, op in enumerate(ready_operations):
        machine_opts = op["machine_options"]
        if not machine_opts:
            scores[i] = -np.inf
            continue

        min_dur = min(machine_opts.values())
        rem_work = op.get("remaining_work", 0.0)
        rem_ops = op.get("remaining_ops", 1) # Avoid division by zero

        # Calculate components
        spt_score = 1.0 / (min_dur + 1e-9)
        mwr_score = rem_work
        finish_score = 1.0 / (rem_ops + 1e-9)

        # Combine
        scores[i] = _W_SPT * spt_score + _W_MWR * mwr_score + _W_FINISH * finish_score

    # Pick the operation with highest score
    idx = int(np.argmax(scores))
    return idx
# EVOLVE-BLOCK-END
