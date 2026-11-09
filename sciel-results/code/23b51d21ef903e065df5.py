"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time

_START_TIME = _time.time()
_DEADLINE = _START_TIME + 115.0  # safety margin before 120s kill


def dispatch(ready_operations, time):
    """
    Dispatch rule combining critical-path urgency with machine-contention-aware routing.
    
    Structural idea: Select the operation with the highest "remaining critical path
    length" (proxy: remaining_work / remaining_ops) to ensure the bottleneck job is
    never delayed. Blend this with a throughput component (1/duration) that becomes
    more important as the schedule progresses. Machine selection applies a soft
    congestion penalty to avoid piling operations onto a single fast machine.
    """
    # Deadline guard
    if _time.time() > _DEADLINE:
        return 0

    if not ready_operations:
        return 0

    n = len(ready_operations)

    # Pre-compute per-operation features
    rws = []
    min_durs = []
    machine_options_list = []
    remaining_ops_counts = []
    critical_paths = []

    # Machine contention: count how many ready ops are eligible for each machine
    machine_usage_counts = {}

    for op in ready_operations:
        mw = op.get("remaining_work", 0)
        ro = op.get("remaining_ops", 0)
        mo = op.get("machine_options", {})

        rws.append(mw)
        remaining_ops_counts.append(ro)
        machine_options_list.append(mo)

        # Critical path length: average remaining work per remaining operation
        # This is a proxy for the minimum processing time chain ahead
        if ro > 0:
            cp = mw / ro
        else:
            cp = mw

        critical_paths.append(cp)

        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            min_durs.append(mo[best_m])
            for m in mo.keys():
                machine_usage_counts[m] = machine_usage_counts.get(m, 0) + 1
        else:
            min_durs.append(0)

    if n == 1:
        op = ready_operations[0]
        mo = op.get("machine_options", {})
        if mo:
            best_m = min(mo.keys(), key=lambda m: mo[m])
            return (0, best_m)
        return 0

    # Phase indicator: how much work is left in the system relative to the max.
    avg_rw = sum(rws) / n
    max_rw = max(rws) if rws else 1.0

    if max_rw > 0:
        phase_ratio = avg_rw / max_rw
    else:
        phase_ratio = 1.0

    # alpha = 1.0 means "critical path urgency", alpha = 0.0 means "throughput"
    alpha = phase_ratio

    # Scale factor for throughput component
    max_cp = max(critical_paths) if critical_paths else 1.0
    if max_cp <= 0:
        max_cp = 1.0

    best_score = -1.0
    best_index = 0
    best_machine = None

    for i in range(n):
        rw = rws[i]
        ro = remaining_ops_counts[i]
        mo = machine_options_list[i]
        cp = critical_paths[i]

        if not mo:
            continue

        # --- Machine selection with congestion penalty ---
        best_m_for_op = None
        best_machine_cost = float('inf')

        for m, d in mo.items():
            cnt = machine_usage_counts.get(m, 0)
            # Soft penalty: discourages choosing a machine that many other ops also want
            penalty = 0.3 * cnt / (1.0 + 0.5 * cnt)
            cost = d + penalty
            if cost < best_machine_cost:
                best_machine_cost = cost
                best_m_for_op = m

        if best_m_for_op is None:
            continue

        min_dur = mo[best_m_for_op]
        if min_dur <= 0:
            min_dur = 1e-9

        # --- Scoring: blend critical path urgency and throughput ---
        # Critical path component: prioritize jobs with long remaining chains
        cp_score = cp

        # Throughput component: inverse of duration (clear the queue)
        srpt_score = 1.0 / (min_dur + 1.0)

        # Scale srpt_score to be comparable to cp_score
        srpt_scaled = srpt_score * (max_cp + 1.0)

        # Blend
        score = alpha * cp_score + (1.0 - alpha) * srpt_scaled

        # Small tie-breaker: jobs with more remaining ops have longer chains
        score += 0.05 * ro

        if score > best_score:
            best_score = score
            best_index = i
            best_machine = best_m_for_op

    return (best_index, best_machine)
# EVOLVE-BLOCK-END
