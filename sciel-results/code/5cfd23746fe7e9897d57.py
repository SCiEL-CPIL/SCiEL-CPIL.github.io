"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np
from math import log, exp

_DEADLINE = 110.0  # Safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.
    
    Strategy:
    1. Calculate 'Urgency' for each ready operation based on remaining work and remaining ops.
    2. Calculate 'Regret' for each operation: the difference between its best and second-best
       machine assignment costs. High regret means the machine choice is critical.
    3. Calculate 'Cost' for each operation-machine pair, penalizing machines with high contention.
    4. Select the operation that maximizes a combination of Urgency and Regret, weighted by
       the inverse of the best cost. This prioritizes critical decisions where the machine
       choice has the highest potential impact on the schedule.
    5. Select the machine that minimizes the Cost for that operation.
    """
    if _elapsed() > _DEADLINE:
        # Hard safety fallback: Shortest Processing Time (SPT)
        if not ready_operations:
            return 0
        best_i = 0
        best_d = float('inf')
        for i, op in enumerate(ready_operations):
            opts = op["machine_options"]
            if not opts:
                continue
            min_d = min(opts.values())
            if min_d < best_d:
                best_d = min_d
                best_i = i
        return best_i

    if not ready_operations:
        return 0
    if len(ready_operations) == 1:
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0

    # Precompute machine contention: count how many ready ops can use each machine
    machine_count = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_count[m] = machine_count.get(m, 0) + 1

    best_score = -float('inf')
    best_index = 0
    best_machine = None

    # Tuning parameters
    CONTENTION_WEIGHT = 0.3
    REMAINING_OPS_WEIGHT = 0.5
    REGRET_WEIGHT = 0.8  # Weight for regret in the final score

    for i, op in enumerate(ready_operations):
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 1.0))
        
        # If no remaining work, skip (shouldn't happen for ready ops, but safe)
        if rw <= 0:
            continue

        # Urgency: Combine remaining work and inverse remaining ops
        # High remaining work = high priority.
        # Low remaining ops = high priority (closer to finish, frees downstream).
        urgency = rw * (1.0 + REMAINING_OPS_WEIGHT / max(ro, 1.0))

        # Calculate costs for all machines for this operation
        costs = []
        machines = []
        
        for m, d in op["machine_options"].items():
            count = machine_count.get(m, 1)
            factor = 1.0 + CONTENTION_WEIGHT * (count - 1)
            cost = float(d) * factor
            costs.append(cost)
            machines.append(m)
        
        if not costs:
            continue
            
        # Sort costs to find best and second-best
        costs_sorted = sorted(costs)
        best_cost = costs_sorted[0]
        
        # Regret: difference between best and second-best
        if len(costs_sorted) > 1:
            second_best_cost = costs_sorted[1]
            regret = second_best_cost - best_cost
        else:
            regret = 0.0

        if best_cost == 0:
            continue

        # Score: Combine urgency and regret, normalized by best cost
        # We want high urgency and high regret (critical decision)
        # The division by best_cost normalizes for the scale of the operation
        score = (urgency + REGRET_WEIGHT * regret) / (best_cost + 1e-9)
        
        if score > best_score:
            best_score = score
            best_index = i
            # Find the machine corresponding to the best cost
            best_machine = machines[costs.index(best_cost)]

    # Fallback if no valid score was found
    if best_machine is None:
        best_index = 0
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            best_machine = min(opts, key=lambda k: opts[k])
        else:
            best_machine = None

    if best_machine is not None:
        return (best_index, best_machine)
    return best_index
# EVOLVE-BLOCK-END
