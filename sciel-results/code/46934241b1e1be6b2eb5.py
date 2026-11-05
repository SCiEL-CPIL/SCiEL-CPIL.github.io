"""Open-Shop Scheduling dispatch rule using a contention-aware scoring approach."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_START_TIME = _time.time()
_DEADLINE = 120.0
_SAFETY_MARGIN = 5.0


def _elapsed() -> float:
    return _time.time() - _START_TIME


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next in an open-shop simulation.
    
    Returns (index, machine) tuple.
    """
    n = len(ready_operations)
    if n == 0:
        return 0
    
    # Deadline guard
    if _elapsed() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: Max remaining work, Min duration
        best_idx = 0
        best_rw = -1
        for i, op in enumerate(ready_operations):
            rw = op["remaining_work"]
            if rw > best_rw:
                best_rw = rw
                best_idx = i
        op = ready_operations[best_idx]
        durations = op["machine_options"]
        best_m = min(durations, key=lambda m: durations[m]) if durations else 0
        return (best_idx, best_m)

    if n == 1:
        op = ready_operations[0]
        durations = op["machine_options"]
        best_m = min(durations, key=lambda m: durations[m]) if durations else 0
        return (0, best_m)

    # 1. Calculate contention for each machine
    machine_contention = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_contention[m] = machine_contention.get(m, 0) + 1

    # Parameters
    LAMBDA = 0.9       # weight of effective duration penalty
    BETA = 0.6         # weight of contention factor
    GAMMA = 0.5        # weight of remaining_ops bonus
    DELTA = 0.3        # weight of max_duration penalty
    SOFT_MIN_T = 5.0   # temperature for soft-min (lower = harder min)

    best_score = -float('inf')
    best_idx = 0
    best_machine = 0

    for i, op in enumerate(ready_operations):
        rw = float(op["remaining_work"])
        rem_ops = float(op.get("remaining_ops", 1))
        durations = op["machine_options"]
        
        if not durations:
            continue

        durs = list(durations.values())
        max_dur = max(durs)
        
        # Find the best machine using soft-min for smoothness
        # cost(m) = duration(m) * (1 + beta * contention(m))
        costs = {}
        for m, d in durations.items():
            cont = machine_contention.get(m, 0)
            costs[m] = d * (1.0 + BETA * cont)
        
        # Soft-min: -T * log(sum(exp(-cost/T))) approximates min(cost)
        # For numerical stability, use the log-sum-exp trick
        cost_vals = list(costs.values())
        min_cost_val = min(cost_vals)
        # Soft-min is between min and mean; use it as a smooth surrogate
        # For simplicity, use a weighted combination: 0.7*min + 0.3*mean
        mean_cost_val = sum(cost_vals) / len(cost_vals)
        min_cost = 0.7 * min_cost_val + 0.3 * mean_cost_val
        
        # Find the machine that achieves the minimum cost (for assignment)
        best_m_for_op = min(costs, key=lambda m: costs[m])
        
        # Calculate score for this operation
        # Score = rw + GAMMA * rem_ops - LAMBDA * min_cost - DELTA * max_dur
        score = rw + GAMMA * rem_ops - LAMBDA * min_cost - DELTA * max_dur
        
        if score > best_score:
            best_score = score
            best_idx = i
            best_machine = best_m_for_op

    return (best_idx, best_machine)
# EVOLVE-BLOCK-END
