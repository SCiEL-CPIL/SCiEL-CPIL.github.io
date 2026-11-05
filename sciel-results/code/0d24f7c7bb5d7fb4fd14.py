"""Job-shop dispatch rule: LRPT with smooth machine-contention and urgency-aware scoring."""
# EVOLVE-BLOCK-START
import time as _time
import math

_dispatch_start_time = _time.time()
_DEADLINE_MARGIN = 140.0


def _deadline_exceeded():
    return (_time.time() - _dispatch_start_time) > _DEADLINE_MARGIN


def dispatch(ready_operations, time):
    """
    Pick which ready operation to start next.
    
    Strategy:
    1. Primary: Maximize Remaining Work (LRPT / Critical Path Length).
    2. Smooth contention penalty using log-sum-exp over machine options to approximate
       the minimum cost in a differentiable way, weighted by machine load.
    3. Urgency bonus: slight preference for jobs with fewer remaining operations
       to reduce the number of pending operations on the critical path.
    4. Return (index, machine) for optimal machine assignment.
    """
    if not ready_operations:
        return 0
    if _deadline_exceeded():
        return 0

    n = len(ready_operations)
    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            return (0, best_m)
        return 0

    # Count how many ready operations target each machine
    machine_counts = {}
    for op in ready_operations:
        mo = op.get("machine_options", {})
        for m in mo:
            machine_counts[m] = machine_counts.get(m, 0) + 1

    best_idx = 0
    best_machine = None
    best_score = float("-inf")

    # Weights: ratio-based urgency (rw/rem_ops) as primary, cost as secondary
    W_RATIO = 1.0
    W_COST = 0.4
    W_URG = 0.05

    # Log-sum-exp temperature for smooth min (lower = sharper, closer to true min)
    T = 1.0

    for i in range(n):
        op = ready_operations[i]
        rw = float(op.get("remaining_work", 0) or 0)
        rem_ops = int(op.get("remaining_ops", 1) or 1)
        if rem_ops < 1:
            rem_ops = 1

        # Ratio-based urgency: work per remaining operation
        ratio = rw / rem_ops

        mo = op.get("machine_options", {})
        if not mo:
            score = W_RATIO * ratio - W_URG * rem_ops
            if score > best_score:
                best_score = score
                best_idx = i
                best_machine = None
            continue

        # Compute smooth min cost over machine options using log-sum-exp
        # cost_m = d * (1 + 0.3 * (count - 1))
        costs = []
        for m, d in mo.items():
            c = machine_counts.get(m, 1)
            cost = d * (1.0 + 0.3 * (c - 1))
            costs.append(cost)

        # Soft-min via log-sum-exp: softmin(x) = -T * log(sum(exp(-x/T)))
        # Normalize by max cost for numerical stability
        max_cost = max(costs) if costs else 1.0
        if max_cost < 1e-9:
            max_cost = 1.0
        norm_costs = [c / max_cost for c in costs]

        max_neg = max([-c / T for c in norm_costs]) if norm_costs else 0.0
        exp_sum = sum(math.exp(-c / T - max_neg) for c in norm_costs)
        soft_min_cost = -T * (math.log(exp_sum) + max_neg) * max_cost

        # Score: High ratio (urgency) is good, Low Cost is good, Low remaining ops is good
        score = W_RATIO * ratio - W_COST * soft_min_cost - W_URG * rem_ops

        if score > best_score:
            best_score = score
            best_idx = i
            # Choose the machine that minimizes the raw cost (not soft-min)
            chosen_m = min(mo.keys(), key=lambda m: mo[m] * (1.0 + 0.3 * (machine_counts.get(m, 1) - 1)))
            best_machine = chosen_m

    if best_machine is not None:
        return (int(best_idx), best_machine)
    else:
        return int(best_idx)

# EVOLVE-BLOCK-END
