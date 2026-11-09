import numpy as np
import time as _time

# Global deadline guard: keep a safety margin before the 120 s evaluator limit.
_DEADLINE = _time.time() + 115.0


def dispatch(ready_operations, time):
    global _DEADLINE
    n = len(ready_operations)
    if n == 0:
        return 0
    if n == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            best_m, best_d = min(opts.items(), key=lambda x: x[1])
            return (0, best_m)
        return 0

    # Fast path: if we're close to the deadline, fall back to a cheap rule.
    if _time.time() > _DEADLINE - 2.0:
        best_i = 0
        best_r = -1.0
        for i, op in enumerate(ready_operations):
            r = op.get("remaining_work", 0.0)
            if r > best_r:
                best_r = r
                best_i = i
        op = ready_operations[best_i]
        opts = op.get("machine_options", {})
        if opts:
            best_m = min(opts, key=opts.get)
            return (best_i, best_m)
        return best_i

    scores = np.zeros(n)
    best_machines = [None] * n

    # Precompute normalization constants
    max_rem_work = max(op["remaining_work"] for op in ready_operations)
    if max_rem_work == 0:
        max_rem_work = 1.0

    for i, op in enumerate(ready_operations):
        rem_work = op["remaining_work"]
        rem_ops = op["remaining_ops"]
        machine_options = op["machine_options"]

        if not machine_options:
            scores[i] = rem_work
            best_machines[i] = None
            continue

        # Bipartite edge weights: w_{i,m} = -rem_work / d_{i,m}
        # We want the operation with the highest MAXIMUM edge weight,
        # i.e., the most negative w (smallest d for large rem_work).
        # Equivalently, maximize  rem_work / min_d  (critical-path density).
        min_dur = float('inf')
        best_m = None
        n_options = 0
        for m, d in machine_options.items():
            n_options += 1
            if d < min_dur:
                min_dur = d
                best_m = m

        if min_dur <= 0 or min_dur == float('inf'):
            scores[i] = rem_work
            best_machines[i] = best_m
            continue

        # Primary metric: critical path per unit machine time on best machine
        # This is the max over machines of (rem_work / d_m), which equals
        # rem_work / min_d since rem_work > 0.
        critical_density = rem_work / min_dur

        # Normalized versions for combining
        norm_work = rem_work / max_rem_work
        norm_density = critical_density / (max_rem_work / min_dur) if min_dur > 0 else 0.0

        # Flexibility: fewer options = more constrained = higher priority
        flexibility = 1.0 / n_options if n_options > 0 else 0.0

        # Heuristic Score combining:
        # - Critical path density (primary): high rem_work / min_d means this op
        #   reduces the critical path the most per unit of machine time.
        # - Normalized remaining work: overall critical path length.
        # - Flexibility: constrained ops (fewer machine choices) should go first.
        # - Remaining ops count: more work left means higher urgency.
        score = (
            2.5 * norm_density
            + 1.8 * norm_work
            + 1.2 * flexibility
            + 0.3 * rem_ops
        )

        scores[i] = score
        best_machines[i] = best_m

    best_idx = int(np.argmax(scores))
    return (best_idx, best_machines[best_idx])
