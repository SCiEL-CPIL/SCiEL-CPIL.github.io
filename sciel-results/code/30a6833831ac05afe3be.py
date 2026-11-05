"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 110.0  # safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def dispatch(ready_operations, time):
    if _elapsed() > _DEADLINE:
        if not ready_operations:
            return 0
        # Fallback: pick op with max remaining work
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    if not ready_operations:
        return 0
    
    if len(ready_operations) == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0

    n = len(ready_operations)

    # 1. Calculate contention for each machine
    # How many ready operations can use each machine?
    machine_contention = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_contention[m] = machine_contention.get(m, 0) + 1

    # 2. Score each operation
    scores = np.empty(n, dtype=float)
    best_machines = [None] * n

    # Weights
    # High weight on remaining work (critical path proxy)
    # Secondary criticality from remaining operation count
    # Penalty on (soft-min) processing time + contention
    w_crit = 2.0
    w_ops = 0.25
    w_cost = 1.0
    lambda_cont = 0.5
    # Temperature for the soft-min (log-sum-exp) surrogate; smaller = closer to hard min
    T = 1.0

    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            scores[i] = -np.inf
            continue

        rem_work = float(op.get("remaining_work", 0.0))
        rem_ops = float(op.get("remaining_ops", 0.0))

        # Per-machine costs
        costs = np.empty(len(opts), dtype=float)
        ms = list(opts.keys())
        for j, m in enumerate(ms):
            cont = machine_contention.get(m, 0)
            costs[j] = float(opts[m]) + lambda_cont * cont

        # Soft-min surrogate via log-sum-exp (smooth, differentiable min)
        # soft_min = -T * log(sum(exp(-cost/T)))
        neg = -costs / T
        m_neg = np.max(neg)
        soft_min = -T * (m_neg + np.log(np.sum(np.exp(neg - m_neg))))

        # Hard-argmin machine for the return value (consistent with the smooth cost)
        best_j = int(np.argmin(costs))
        best_m = ms[best_j]

        scores[i] = w_crit * rem_work + w_ops * rem_ops - w_cost * soft_min
        best_machines[i] = best_m

    # 3. Pick the best operation
    best_idx = int(np.argmax(scores))
    
    # Deadline check
    if _elapsed() > _DEADLINE:
        return int(np.argmax([op["remaining_work"] for op in ready_operations]))

    # Return index and best machine
    if best_machines[best_idx] is not None:
        return (best_idx, best_machines[best_idx])
    return best_idx
# EVOLVE-BLOCK-END
