"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
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
       Jobs with more remaining work are critical, but jobs with fewer remaining ops
       are closer to completion and should be prioritized to clear the queue.
    2. Calculate 'Cost' for each operation-machine pair using a smooth surrogate
       (soft-min via log-sum-exp) to balance duration and machine contention.
    3. Select the operation with the highest Urgency/Cost ratio.
    4. Select the machine that minimizes the expected cost for that operation.
    """
    if _elapsed() > _DEADLINE:
        # Hard safety fallback: Shortest Processing Time (SPT) on best available machine
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
    CONTENTION_WEIGHT = 0.4  # Increased slightly to penalize congestion more
    REMAINING_OPS_WEIGHT = 0.5  # Weight for remaining ops in urgency
    SOFT_MIN_TEMPERATURE = 0.5  # Temperature for soft-min (lower = sharper min)

    for i, op in enumerate(ready_operations):
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 1.0))
        
        # If no remaining work, skip (shouldn't happen for ready ops, but safe)
        if rw <= 0:
            continue

        # Urgency: Combine remaining work and inverse remaining ops
        # High remaining work = high priority.
        # Low remaining ops = high priority (closer to finish, frees downstream).
        # We use rw * (1 + REMAINING_OPS_WEIGHT / ro) to boost jobs near completion.
        urgency = rw * (1.0 + REMAINING_OPS_WEIGHT / max(ro, 1.0))

        # Find the best machine for this operation considering contention
        # Using a smooth surrogate for the cost to avoid brittle hard-mins
        costs = []
        machines = []
        
        for m, d in op["machine_options"].items():
            count = machine_count.get(m, 1)
            # Base cost is duration adjusted by contention
            # factor = 1.0 + CONTENTION_WEIGHT * (count - 1)
            # cost = float(d) * factor
            
            # We want to select the machine that minimizes this cost.
            # To use a soft-min for the overall score calculation, we first
            # compute the raw costs.
            factor = 1.0 + CONTENTION_WEIGHT * (count - 1)
            cost = float(d) * factor
            costs.append(cost)
            machines.append(m)

        if not costs:
            continue

        # Apply soft-min to get a smooth cost value for the operation
        # soft_min(x, t) = -t * log(sum(exp(-x_i / t)))
        # As t -> 0, this approaches min(x_i).
        # We use this smooth value as the denominator in the score.
        
        # Numerical stability for log-sum-exp
        min_cost_val = min(costs)
        if min_cost_val < 1e-6:
            # If costs are tiny, just use the min to avoid log(0) issues or large exps
            smooth_cost = min_cost_val
        else:
            # Scale costs to avoid overflow in exp
            # We are computing -t * log(sum(exp(-c_i / t)))
            # Let y_i = -c_i / t
            # We need log(sum(exp(y_i)))
            # max_y = max(y_i) = -min(c_i)/t
            # sum(exp(y_i)) = exp(max_y) * sum(exp(y_i - max_y))
            # log(sum(exp(y_i))) = max_y + log(sum(exp(y_i - max_y)))
            
            t = SOFT_MIN_TEMPERATURE
            # y_i = -c_i / t
            # max_y = -min_cost_val / t
            max_y = -min_cost_val / t
            
            sum_exp = 0.0
            for c in costs:
                y = -c / t
                # y - max_y = (-c/t) - (-min_cost_val/t) = (min_cost_val - c) / t
                # Since c >= min_cost_val, (min_cost_val - c) <= 0, so exp is <= 1. Safe.
                sum_exp += exp((min_cost_val - c) / t)
            
            # log(sum(exp(y))) = max_y + log(sum_exp)
            log_sum_exp = max_y + log(sum_exp)
            
            # soft_min = -t * log_sum_exp
            smooth_cost = -t * log_sum_exp

        if smooth_cost <= 0:
            smooth_cost = 1e-9

        # Score: High urgency, low cost.
        score = urgency / (smooth_cost + 1e-9)
        
        if score > best_score:
            best_score = score
            best_index = i
            # For the actual machine selection, we still pick the one with the lowest
            # raw cost (hard min) because the simulation needs a concrete machine.
            # The soft-min was used to smooth the *score* calculation.
            min_c = float('inf')
            best_m_for_op = None
            for m, c in zip(machines, costs):
                if c < min_c:
                    min_c = c
                    best_m_for_op = m
            best_machine = best_m_for_op

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
