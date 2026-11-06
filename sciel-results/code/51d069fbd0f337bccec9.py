"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time

_DEADLINE = 110.0  # safety margin below the 150s wall-clock kill
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
    1. Urgency: Prioritize operations from jobs with more remaining operations
       (proxy for critical path length). This ensures jobs that are "deep" in their
       sequence are completed, freeing up subsequent resources.
    2. Machine Selection: Select the machine that minimizes the weighted sum of
       the operation's duration and the current load of other ready operations
       on that machine. This balances the load across machines.
    3. Score: Maximize (Urgency / Effective_Cost). This is a Weighted Shortest 
       Job First (WSJF) variant adapted for flexible machines.
    """
    if _elapsed() > _DEADLINE:
        # Hard safety fallback: Shortest Processing Time (SPT)
        if not ready_operations:
            return 0
        best_i = 0
        best_d = float('inf')
        best_m = None
        for i, op in enumerate(ready_operations):
            opts = op["machine_options"]
            if not opts:
                continue
            min_d = min(opts.values())
            if min_d < best_d:
                best_d = min_d
                best_i = i
                best_m = min(opts, key=lambda k: opts[k])
        if best_m is not None:
            return (best_i, best_m)
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

    n = len(ready_operations)

    # Precompute remaining ops (urgency) and machine options for all ready ops
    # We use remaining_ops as urgency because it reflects the depth of the job
    # remaining in the schedule, which is a better proxy for critical path than
    # just remaining_work (which can be skewed by one long operation).
    
    # Precompute machine load: total committed duration of all ready ops per machine.
    # This allows us to estimate future congestion.
    machine_load = {}
    op_data = []
    
    for op in ready_operations:
        ro = float(op.get("remaining_ops", 1.0))
        opts = op.get("machine_options", {})
        op_data.append((ro, opts))
        
        for m, d in opts.items():
            machine_load[m] = machine_load.get(m, 0.0) + float(d)

    # Weight for load penalty. 
    # A higher value makes the rule more aggressive in avoiding congested machines.
    # 0.5 is a balanced choice: it considers load but doesn't ignore the actual processing time.
    LOAD_WEIGHT = 0.5

    best_score = -float('inf')
    best_index = 0
    best_machine = None
    best_cost = float('inf')

    for i in range(n):
        ro, opts = op_data[i]
        
        if not opts:
            continue
            
        # Urgency: Higher is better. 
        # We use remaining_ops directly. 
        # If a job has 10 ops left, it's more critical to start it than one with 1 op left,
        # assuming similar processing times, because it has more downstream dependencies.
        urgency = ro
        
        if urgency <= 0:
            urgency = 1e-9

        # Find the best machine for this operation
        best_m_for_op = None
        min_cost_for_op = float('inf')

        for m, d in opts.items():
            d = float(d)
            if d <= 0:
                d = 1e-9
            
            # Load of other operations on this machine
            other_load = machine_load.get(m, 0.0) - d
            
            # Effective cost: duration + penalty for existing load
            cost = d + LOAD_WEIGHT * other_load
            
            if cost < min_cost_for_op:
                min_cost_for_op = cost
                best_m_for_op = m

        if best_m_for_op is None:
            continue

        # Score: Urgency / Cost
        # We want to maximize this. High urgency (many ops left) and low cost (fast machine, low load)
        score = urgency / min_cost_for_op
        
        # Tie-breaking: If scores are very close, prefer the one with lower absolute cost
        # This helps in breaking ties between operations with similar urgency but different durations.
        if score > best_score + 1e-9 or (abs(score - best_score) <= 1e-9 and min_cost_for_op < best_cost):
            best_score = score
            best_index = i
            best_machine = best_m_for_op
            best_cost = min_cost_for_op

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
