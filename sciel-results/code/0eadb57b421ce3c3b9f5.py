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
    1. Urgency: prioritize jobs with the most remaining work and remaining ops
       (avoids starving long jobs and keeps the schedule progressing).
    2. Machine selection: for the chosen operation, pick the eligible machine
       that minimizes  duration + W * backlog,  where backlog is the committed
       load of the *other* ready operations that can also use that machine.
       This balances short processing time against avoiding machine bottlenecks.
    3. Regret-based selection: prioritize operations where the machine choice
       has the highest potential impact (max difference between best and
       second-best machine costs), combined with urgency.
    4. Tie-breaking: prefer the machine with the least total ready-op
       contention (spreads load across machines).

    The combination favors finishing urgent work on the least-loaded machine,
    while prioritizing decisions where machine choice matters most.
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

    n = len(ready_operations)

    # Precompute remaining work, remaining ops, and machine options for all ready ops
    op_data = []
    for op in ready_operations:
        rw = float(op.get("remaining_work", 0.0))
        ro = int(op.get("remaining_ops", 1))
        opts = op.get("machine_options", {})
        op_data.append((rw, ro, opts))

    # Precompute machine load: total committed duration of all ready ops per machine.
    machine_load = {}
    for rw, ro, opts in op_data:
        for m, d in opts.items():
            machine_load[m] = machine_load.get(m, 0.0) + float(d)

    # Congestion weight: how strongly we penalize machines that other ready ops
    # also need. Higher => avoid bottlenecks more aggressively.
    W = 0.5
    # Regret weight: how much we prioritize operations with high machine-choice regret
    REGRET_WEIGHT = 0.3
    # Critical path weight: adds urgency for jobs with more remaining operations
    CRITICAL_PATH_WEIGHT = 0.1

    best_score = -float('inf')
    best_index = 0
    best_machine = None

    for i in range(n):
        rw, ro, opts = op_data[i]
        if rw <= 0:
            continue

        # Compute urgency: remaining work scaled by critical path length
        urgency = rw * (1.0 + CRITICAL_PATH_WEIGHT * ro)

        # Compute machine costs for this operation and find regret
        machine_costs = []
        for m, d in opts.items():
            d = float(d)
            if d == 0:
                continue
            load = machine_load.get(m, d)
            other_load = load - d  # committed load of *other* ready ops on m
            cost = d + W * other_load
            if cost <= 0:
                cost = 1e-9
            machine_costs.append((cost, m, load))

        if not machine_costs:
            continue

        # Sort by cost to find best and second-best
        machine_costs.sort(key=lambda x: (x[0], x[2]))

        best_cost, best_m, best_load = machine_costs[0]

        # Regret: difference between best and second-best cost
        if len(machine_costs) > 1:
            second_cost = machine_costs[1][0]
            regret = second_cost - best_cost
        else:
            regret = 0.0

        # Score: urgency / effective cost, boosted by regret
        base_score = urgency / best_cost
        score = base_score * (1.0 + REGRET_WEIGHT * regret / max(best_cost, 1e-9))

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m

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
