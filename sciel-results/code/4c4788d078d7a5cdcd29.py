"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = _time.monotonic() + 145.0  # safety margin before 150s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule for job-shop scheduling.
    Returns index of the ready operation to start next.
    """
    n = len(remaining_ops_list := [op["remaining_ops"] for op in ready_operations])
    if n == 0:
        return 0

    # Compute a composite score for each ready operation.
    # We want to balance:
    # 1. Shortest processing time (SPT) on the chosen machine -> frees machine quickly
    # 2. Most work remaining (MWRT) -> progress long jobs
    # 3. Fewest remaining operations -> finish jobs soon
    # 4. NSPT-like ratio: min_duration / (remaining_work + 1)

    scores = np.empty(n, dtype=np.float64)
    for i, op in enumerate(ready_operations):
        machine_options = op["machine_options"]
        if not machine_options:
            scores[i] = 0.0
            continue

        # Find the minimum duration among machine options
        min_dur = min(machine_options.values())
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]

        # Component 1: NSPT-like score (higher is better for shorter relative duration)
        # Invert so that shorter duration gives higher score
        nspt_score = 1.0 / (min_dur + 1e-9)

        # Component 2: MWRT (higher remaining work -> higher priority)
        mwrt_score = rem_work

        # Component 3: Fewer remaining ops -> higher priority (to finish jobs)
        few_ops_score = 1.0 / (rem_ops + 1e-9)

        # Weighted combination
        # Weights: prioritize NSPT (machine freeing) and MWRT (job progress)
        # Few ops is a tiebreaker
        score = 0.4 * nspt_score + 0.4 * (mwrt_score / (rem_work + 1.0)) + 0.2 * few_ops_score

        scores[i] = score

    # Safety: check deadline
    if _time.monotonic() > _DEADLINE:
        # Fallback: just pick the first one with min duration
        min_durs = []
        for op in ready_operations:
            mo = op["machine_options"]
            min_durs.append(min(mo.values()) if mo else float('inf'))
        return int(np.argmin(min_durs))

    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
