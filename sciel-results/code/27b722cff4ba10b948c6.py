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
    # Moderate penalty on processing time + contention
    w_crit = 2.0
    w_cost = 1.0
    lambda_cont = 0.5
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            scores[i] = -np.inf
            continue
            
        rem_work = float(op.get("remaining_work", 0.0))
        
        # Find the best machine for this operation
        # Cost = duration + lambda * contention
        min_cost = np.inf
        best_m = None
        for m, dur in opts.items():
            cont = machine_contention.get(m, 0)
            cost = float(dur) + lambda_cont * cont
            if cost < min_cost:
                min_cost = cost
                best_m = m
        
        scores[i] = w_crit * rem_work - w_cost * min_cost
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
