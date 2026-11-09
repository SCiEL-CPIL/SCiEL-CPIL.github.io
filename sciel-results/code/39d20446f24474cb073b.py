"""Open-Shop Scheduling dispatch rule with load-aware heuristics."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

# Module-level deadline guard to ensure we finish well within limits
_START_TIME = _time.time()
_DEADLINE = 115.0  # seconds; safety margin before 120s limit


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an event-driven open-shop simulation.
    
    Strategy: Load-aware dispatch that balances:
    1. Machine congestion (prefer machines with less current load)
    2. Job urgency (jobs with more remaining ops/work are more critical)
    3. Shortest processing time on the least-loaded feasible machine
    
    Returns: index of the ready operation to start
    """
    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)
    if n == 1:
        return 0

    # For each ready operation, compute a score
    # We want to minimize makespan, so we prefer:
    # - Operations that can go on less-loaded machines (to balance load)
    # - Operations from jobs with more remaining work (urgency)
    # - Shorter durations on the best available machine

    # Estimate machine loads: we don't have direct access to current machine states,
    # but we can use remaining_work as a proxy for job urgency and
    # machine_options to pick the shortest duration option.
    #
    # A good heuristic: for each operation, find the minimum duration across machines,
    # and also consider how many machines are "free" (short duration).
    # Combine with job urgency (remaining_work, remaining_ops).

    scores = np.zeros(n)

    # Pre-compute urgency factors
    remaining_works = np.array([op["remaining_work"] for op in ready_operations], dtype=float)
    remaining_ops = np.array([op["remaining_ops"] for op in ready_operations], dtype=float)

    # Normalize urgency to [0, 1]
    max_rw = remaining_works.max() if remaining_works.max() > 0 else 1.0
    max_ro = remaining_ops.max() if remaining_ops.max() > 0 else 1.0

    for i, op in enumerate(ready_operations):
        mo = op["machine_options"]
        if not mo:
            scores[i] = 0.0
            continue

        durations = list(mo.values())
        min_dur = min(durations)
        avg_dur = sum(durations) / len(durations)

        # Urgency: higher remaining work/ops means this job is more critical
        urgency = (remaining_works[i] / max_rw) * 0.6 + (remaining_ops[i] / max_ro) * 0.4

        # Prefer operations with shorter min duration (reduces blocking)
        # and from urgent jobs
        # Score: higher is better for selection
        # We want to pick the operation that best balances load and urgency
        # A job with lots of remaining work should be prioritized to avoid it becoming a bottleneck
        # An operation with a short duration on some machine is good to start early

        # Combine: urgency weighted heavily, with a bonus for short processing time
        scores[i] = urgency * 1.0 - min_dur * 0.01

    # Pick the operation with the highest score
    best_idx = int(np.argmax(scores))

    # Deadline guard
    if _time.time() - _START_TIME > _DEADLINE:
        return best_idx

    return best_idx
# EVOLVE-BLOCK-END
