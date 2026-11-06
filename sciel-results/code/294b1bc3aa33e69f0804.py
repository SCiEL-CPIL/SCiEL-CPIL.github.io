"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import numpy as np

_DEADLINE = 115.0  # safety margin below the 150s wall-clock kill
_T0 = None


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.
    
    Strategy: Regret-Based Lookahead
    1. For each ready operation, estimate the completion time if scheduled
       on each eligible machine, accounting for current machine contention.
    2. Calculate the 'regret' for each operation: the difference between the
       cost of using its best machine and its second-best machine.
    3. Select the operation with the highest regret. This prioritizes jobs
       that would suffer most from machine unavailability or congestion.
    4. Assign the selected operation to its best machine.
    
    This structural change shifts from greedy individual-job prioritization
    to a contention-aware opportunity-cost model, better handling the
    coupling between concurrent ready operations.
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
            # Pick the machine with the shortest duration
            m = min(opts, key=lambda k: opts[k])
            return (0, m)
        return 0

    n = len(ready_operations)
    
    # 1. Precompute machine contention
    # Count how many ready operations can use each machine
    machine_count = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_count[m] = machine_count.get(m, 0) + 1

    # 2. Calculate estimated cost for each (operation, machine) pair
    # Cost = duration + contention_penalty
    # Contention penalty is proportional to the number of other ops wanting that machine
    # We use a soft penalty to avoid hard blocking but reflect congestion
    # penalty = alpha * (count - 1)
    alpha = 0.5
    
    # Store best and second-best costs for each operation
    best_costs = np.full(n, float('inf'))
    second_best_costs = np.full(n, float('inf'))
    best_machines = [None] * n
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            continue
            
        local_best = float('inf')
        local_second = float('inf')
        local_best_m = None
        
        for m, d in opts.items():
            d = float(d)
            count = machine_count.get(m, 1)
            # Estimate cost: duration + penalty for contention
            # If count is 1, penalty is 0. If count is 2, penalty is alpha.
            penalty = alpha * (count - 1)
            cost = d + penalty
            
            if cost < local_best:
                local_second = local_best
                local_best = cost
                local_best_m = m
            elif cost < local_second:
                local_second = cost
        
        best_costs[i] = local_best
        second_best_costs[i] = local_second
        best_machines[i] = local_best_m

    # 3. Calculate Regret
    # Regret = second_best_cost - best_cost
    # If an operation has only one machine option, regret is 0 (no choice, but also no loss if delayed? 
    # Actually, if it has only one machine, its "regret" of not scheduling is 0 in this metric, 
    # but it might be urgent. However, standard regret heuristics often focus on the difference.
    # To handle single-machine ops, we might add a small urgency term or just rely on the cost.
    # Let's stick to pure regret for now, as it drives the structural change.
    # However, if all regrets are 0 (e.g., all single-machine), we need a tie-breaker.
    # Tie-breaker: lowest best_cost (SPT-like) or highest remaining work?
    # Let's use lowest best_cost as tie-breaker to favor quick completion of uncontested jobs.
    
    regrets = second_best_costs - best_costs
    
    # Handle cases where second_best is inf (only 1 machine option)
    # In that case, regret is inf? No, inf - finite = inf.
    # If an op has only 1 machine, its regret is effectively "infinite" in the sense that 
    # if you don't schedule it, you can't schedule it on an alternative. 
    # But usually, regret is defined as loss of optimality. 
    # If there's no alternative, the "loss" of not picking the best is 0 because there is no best vs second.
    # Wait, if I don't schedule Op A (1 machine), and I schedule Op B (2 machines), 
    # Op A is stuck. Op B can use its 2nd machine.
    # So Op A should have higher priority?
    # Let's define: if only 1 option, regret = large constant? 
    # Or simply: if second_best is inf, set regret to a high value to prioritize securing the sole resource?
    # Actually, if I have a machine that ONLY one job can use, that job is critical.
    # Let's set regret to 1000.0 if second_best is inf, to ensure it gets picked.
    
    for i in range(n):
        if np.isinf(second_best_costs[i]):
            regrets[i] = 1000.0
        else:
            # Ensure non-negative regret
            regrets[i] = max(0.0, regrets[i])

    # 4. Select operation with max regret
    # If tie, select the one with lowest best_cost (earliest completion)
    max_regret_idx = -1
    max_regret_val = -1.0
    
    for i in range(n):
        if regrets[i] > max_regret_val:
            max_regret_val = regrets[i]
            max_regret_idx = i
        elif regrets[i] == max_regret_val:
            # Tie-breaker: lower best_cost is better
            if best_costs[i] < best_costs[max_regret_idx]:
                max_regret_idx = i

    # Fallback if something went wrong
    if max_regret_idx == -1:
        return 0

    selected_op_idx = max_regret_idx
    selected_machine = best_machines[selected_op_idx]
    
    if selected_machine is None:
        # Fallback to first available machine
        opts = ready_operations[selected_op_idx]["machine_options"]
        if opts:
            selected_machine = list(opts.keys())[0]
        else:
            return selected_op_idx

    return (selected_op_idx, selected_machine)

# EVOLVE-BLOCK-END
