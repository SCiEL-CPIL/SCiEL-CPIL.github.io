"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Explicit deadline guard: evaluator kills after 150s wall-clock.
# We track elapsed time and return early if we approach the limit.
_START_TIME = _time.monotonic()
_DEADLINE = 145.0  # safety margin of 5s


def _elapsed():
    return _time.monotonic() - _START_TIME


def dispatch(ready_operations, time):
    # Fast path: if only one option, return it
    if len(ready_operations) == 1:
        return 0

    # Deadline guard: if we're close to the limit, fall back to a simple rule
    if _elapsed() > _DEADLINE - 1.0:
        # Simple fallback: most remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    n = len(ready_operations)
    jobs = [op["job"] for op in ready_operations]
    ops = [op["op"] for op in ready_operations]
    remaining_works = [op["remaining_work"] for op in ready_operations]
    remaining_ops = [op["remaining_ops"] for op in ready_operations]
    machine_options_list = [op["machine_options"] for op in ready_operations]

    # For each ready operation, compute the minimum duration across machines
    min_durations = []
    for mo in machine_options_list:
        if mo:
            min_durations.append(min(mo.values()))
        else:
            min_durations.append(0)

    # Multi-criteria dispatch:
    # 1. Longest remaining work (LRW) - primary
    # 2. Shortest processing time (SPT) on the best machine - secondary
    # 3. Most remaining operations - tertiary
    # Combine into a single score using weighted sum with normalization

    # Normalize each criterion to [0, 1] range
    max_rw = max(remaining_works) if remaining_works else 1
    min_rw = min(remaining_works) if remaining_works else 0
    rw_range = max_rw - min_rw if max_rw > min_rw else 1

    max_md = max(min_durations) if min_durations else 1
    min_md = min(min_durations) if min_durations else 0
    md_range = max_md - min_md if max_md > min_md else 1

    max_ro = max(remaining_ops) if remaining_ops else 1
    min_ro = min(remaining_ops) if remaining_ops else 0
    ro_range = max_ro - min_ro if max_ro > min_ro else 1

    scores = []
    for i in range(n):
        # Normalize remaining work (higher is better)
        rw_norm = (remaining_works[i] - min_rw) / rw_range
        # Normalize min duration (lower is better, so invert)
        md_norm = 1.0 - (min_durations[i] - min_md) / md_range
        # Normalize remaining ops (higher is better)
        ro_norm = (remaining_ops[i] - min_ro) / ro_range

        # Weighted combination: emphasize remaining work most
        score = 0.5 * rw_norm + 0.3 * md_norm + 0.2 * ro_norm
        scores.append(score)

    # Return index of the operation with the highest combined score
    return int(np.argmax(scores))
# EVOLVE-BLOCK-END
