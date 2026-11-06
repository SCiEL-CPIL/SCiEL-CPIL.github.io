"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math

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
    
    Strategy: WSJF (Weighted Shortest Job First) variant with load-penalized cost.
    
    1. For each ready operation, compute:
       - weight = remaining_work / max(1, remaining_ops)  (urgency: avg work per remaining op)
       - For each machine option, compute a load-penalized processing time:
         cost_m = duration * (1 + alpha * (num_ready_ops_using_m - 1))
       - The effective "job length" is the minimum cost_m across machines.
    2. Score each operation as: score = weight / effective_length
       Higher score means higher priority (more urgent, shorter effective job).
    3. Select the operation with the highest score.
    4. Assign it to the machine with the lowest load-penalized cost.
    
    This is a well-established heuristic that balances job criticality (weight)
    and machine load (penalized duration), typically outperforming complex
    regret-based surrogates on standard FJSP instances.
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
            min_d = min(float(d) for d in opts.values())
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
            m = min(opts, key=lambda k: float(opts[k]))
            return (0, m)
        return 0

    n = len(ready_operations)
    
    # Precompute machine contention: count how many ready ops can use each machine
    machine_count = {}
    for op in ready_operations:
        for m in op["machine_options"]:
            machine_count[m] = machine_count.get(m, 0) + 1
    
    # Load penalty weight
    ALPHA = 0.5
    
    best_score = -float('inf')
    best_index = 0
    best_machine = None
    best_cost = float('inf')
    
    for i, op in enumerate(ready_operations):
        opts = op["machine_options"]
        if not opts:
            continue
        
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 0.0))
        
        # Weight: urgency based on average remaining work per remaining operation
        if ro > 0:
            weight = rw / ro
        else:
            weight = rw if rw > 0 else 1.0
        
        # Ensure weight is positive
        if weight <= 0:
            weight = 1.0
        
        # Compute load-penalized costs for all machines
        min_cost = float('inf')
        best_m = None
        
        for m, d in opts.items():
            d = float(d)
            if d <= 0:
                d = 1.0
            
            count = machine_count.get(m, 1)
            # Load penalty: more contention means higher effective cost
            load_factor = 1.0 + ALPHA * (count - 1)
            cost = d * load_factor
            
            if cost < min_cost:
                min_cost = cost
                best_m = m
        
        if best_m is None:
            continue
        
        # WSJF score: higher weight / shorter effective job = higher priority
        score = weight / min_cost
        
        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m
            best_cost = min_cost
    
    # Fallback if no valid score was found
    if best_machine is None:
        best_index = 0
        op = ready_operations[0]
        opts = op["machine_options"]
        if opts:
            best_machine = min(opts, key=lambda k: float(opts[k]))
        else:
            best_machine = None
    
    if best_machine is not None:
        return (best_index, best_machine)
    return best_index
# EVOLVE-BLOCK-END
