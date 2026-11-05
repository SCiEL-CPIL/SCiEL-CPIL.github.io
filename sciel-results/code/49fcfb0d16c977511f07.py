"""Model-and-optimize dispatch for flexible job-shop scheduling."""
# EVOLVE-BLOCK-START
import time as _time
import math as _math

_DEADLINE = 110.0  # safety margin below the 150s wall-clock kill
_T0 = None


def _exp(x):
    """Clamped exp to avoid overflow in log-sum-exp."""
    if x > 50.0:
        return 1e22
    if x < -50.0:
        return 0.0
    return _math.exp(x)


def _log(x):
    if x <= 0.0:
        return -700.0
    return _math.log(x)


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = _time.perf_counter()
    return _time.perf_counter() - _T0


def dispatch(ready_operations, time):
    """
    Dispatch rule for Flexible Job-Shop Scheduling.

    Strategy:
    1. Urgency: prioritize jobs with the most remaining work (avoids starving
       long jobs and keeps the schedule progressing).
    2. Machine selection: for the chosen operation, pick the eligible machine
       that minimizes  duration + W * backlog,  where backlog is the committed
       load of the *other* ready operations that can also use that machine.
       This balances short processing time against avoiding machine bottlenecks.
    3. Tie-breaking: prefer the machine with the least total ready-op
       contention (spreads load across machines).

    The combination favors finishing urgent work on the least-loaded machine,
    which empirically reduces makespan in flexible job-shop settings.
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

    # Precompute remaining work, remaining ops count, and machine options
    op_data = []
    for op in ready_operations:
        rw = float(op.get("remaining_work", 0.0))
        ro = float(op.get("remaining_ops", 1.0))
        opts = op.get("machine_options", {})
        op_data.append((rw, ro, opts))

    # Precompute machine load: total committed duration of all ready ops per machine.
    machine_load = {}
    for rw, ro, opts in op_data:
        for m, d in opts.items():
            machine_load[m] = machine_load.get(m, 0.0) + float(d)

    # Congestion weight: how strongly we penalize machines that other ready ops
    # also need. Higher => avoid bottlenecks more aggressively.
    W = 0.8

    # Soft-min temperature for smoothing the machine selection (log-sum-exp).
    # Smaller => closer to hard argmin; larger => smoother surrogate.
    T = 0.5

    best_score = -float('inf')
    best_index = 0
    best_machine = None

    for i in range(n):
        rw, ro, opts = op_data[i]
        if rw <= 0:
            continue

        # Find the best machine for this operation using a soft-min surrogate.
        # cost_m = d + W * other_load;  effective_cost = T * logsumexp(cost/T)
        # This smooths the nonsmooth min over machines, which can help the
        # outer urgency/cost ratio behave more continuously.
        costs = []
        for m, d in opts.items():
            d = float(d)
            if d == 0:
                continue
            load = machine_load.get(m, d)
            other_load = load - d
            c = d + W * other_load
            if c <= 0:
                c = 1e-9
            costs.append((c, m, load))

        if not costs:
            continue

        # Soft-min (log-sum-exp) surrogate for the minimum cost
        max_c = max(c for c, _, _ in costs)
        # logsumexp(c/T) = max_c/T + log(sum(exp((c - max_c)/T)))
        s = 0.0
        for c, _, _ in costs:
            s += _exp((c - max_c) / T)
        if s <= 0:
            s = 1e-300
        eff_cost = T * (max_c / T + _log(s))
        if eff_cost <= 0:
            eff_cost = 1e-9

        # Pick the argmin machine as the concrete assignment (tie-break by load)
        best_m_for_op = None
        best_pair_cost = float('inf')
        best_pair_tie = float('inf')
        for c, m, load in costs:
            if c < best_pair_cost or (c == best_pair_cost and load < best_pair_tie):
                best_pair_cost = c
                best_pair_tie = load
                best_m_for_op = m

        # Urgency: combine remaining work (total load) with remaining ops count
        # (path-length proxy). More remaining ops => longer critical path ahead.
        urgency = rw + 0.3 * ro

        # Score: urgency / soft-min effective cost. Higher is better.
        score = urgency / eff_cost

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op

    # Fallback if no valid score was found
    if best_machine is None:
        pass
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
