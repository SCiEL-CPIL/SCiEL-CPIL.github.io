import numpy as np
import time
from collections import deque
import random

_T0 = time.time()
_DEADLINE = 150.0
_MARGIN = 5.0

def _elapsed():
    return time.time() - _T0

def _topo_order(n, preds):
    """Return topological order and successor list."""
    indeg = [0] * n
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            indeg[i] += 1
            succ[p].append(i)
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) < n:  # cycle fallback
        order = list(range(n))
    return order, succ

def _simulate_serial_schedule(eligible_set, preds, succ, durations, demands, capacities, n, R):
    """
    Simulate a complete serial schedule from the current state.
    Returns the makespan.
    """
    scheduled = set()
    remaining = set(range(n))
    resource_usage = np.zeros(R, dtype=float)
    current_time = 0.0
    makespan = 0.0

    # Pre-compute finish times for already-scheduled activities (none in this case)
    finish_times = np.zeros(n, dtype=float)

    while remaining:
        # Find eligible activities: all predecessors scheduled
        eligible = []
        for act in remaining:
            if all(p in scheduled for p in preds[act]):
                # Check if resources are available
                if R > 0 and demands[act].size:
                    if np.all(demands[act] <= capacities - resource_usage + 1e-9):
                        eligible.append(act)
                else:
                    eligible.append(act)

        if not eligible:
            # Should not happen if instance is valid, but handle gracefully
            break

        # Pick the eligible activity with highest tail (critical path heuristic)
        # For MCTS simulation, use a simple deterministic rule: highest tail
        best_act = None
        best_tail = -1
        for act in eligible:
            # Compute a quick tail estimate
            t = durations[act]
            for s in succ[act]:
                if s not in scheduled:
                    # Approximate: just use duration for now (fast)
                    t = max(t, durations[act])
            # Use precomputed tail if available, otherwise duration
            if best_tail < t:
                best_tail = t
                best_act = act

        # Start the best activity at current time
        start_time = current_time
        finish_time = start_time + durations[best_act]

        # Update resource usage
        if R > 0 and demands[best_act].size:
            resource_usage += demands[best_act]

        finish_times[best_act] = finish_time
        scheduled.add(best_act)
        remaining.discard(best_act)
        makespan = max(makespan, finish_time)

        # Advance time to the next event: when any running activity finishes
        # In serial SGS, we advance to max(finish_times of scheduled that are "done")
        # Actually in serial SGS, time advances to the earliest time when a new activity can start
        # Simplified: advance to the max finish time of currently "active" activities
        # Since serial SGS schedules one at a time, we just advance to finish_time
        current_time = finish_time

    return makespan


def priority(instance):
    """
    Score each activity for a serial schedule-generation scheme using MCTS-inspired
    Monte Carlo simulation to estimate which activities lead to shorter makespans.
    Higher score = higher priority (started earlier).
    """
    if _elapsed() > _DEADLINE - _MARGIN:
        return np.asarray(instance["durations"], dtype=float)

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    preds = instance["predecessors"]
    demands = np.asarray(instance["demands"], dtype=float)
    capacities = np.asarray(instance["capacities"], dtype=float)

    if demands.ndim == 1:
        if n > 0:
            demands = demands.reshape(n, -1)
        else:
            demands = demands.reshape(0, 0)
    R = demands.shape[1] if demands.size else 0

    order, succ = _topo_order(n, preds)

    # Compute tail (critical path) for base heuristic
    tail = np.zeros(n, dtype=float)
    for i in reversed(order):
        best = 0.0
        for s in succ[i]:
            if tail[s] > best:
                best = tail[s]
        tail[i] = durations[i] + best

    # Successor count
    succ_count = np.zeros(n, dtype=float)
    for i in range(n):
        succ_count[i] = len(succ[i])

    # Resource pressure
    if R > 0 and capacities.size:
        cap = np.where(capacities <= 0, 1e-9, capacities)
        pressure = np.sum(demands / cap, axis=1)
    else:
        pressure = np.zeros(n, dtype=float)

    # Find currently eligible activities (no predecessors)
    eligible_set = set()
    for i in range(n):
        if len(preds[i]) == 0:
            eligible_set.add(i)

    if not eligible_set:
        eligible_set = {0} if n > 0 else set()

    # Monte Carlo simulation: for each eligible activity, simulate multiple
    # completions starting with that activity to estimate expected makespan
    sim_counts = min(200, max(50, int((30.0 - _elapsed()) * 10)))
    if _elapsed() > _DEADLINE - _MARGIN:
        sim_counts = 10

    makespan_sums = np.zeros(n, dtype=float)
    makespan_counts = np.zeros(n, dtype=float)

    eligible_list = list(eligible_set)
    random.seed(42)

    for act in eligible_list:
        if _elapsed() > _DEADLINE - _MARGIN:
            break

        for _ in range(sim_counts):
            # Simulate: force 'act' to start first, then use critical-path heuristic
            scheduled = {act}
            remaining = set(range(n)) - {act}
            resource_usage = np.zeros(R, dtype=float)
            if R > 0 and demands[act].size:
                resource_usage += demands[act]
            finish_times = np.zeros(n, dtype=float)
            finish_times[act] = durations[act]
            current_time = durations[act]
            makespan = durations[act]

            while remaining:
                # Find eligible activities
                eligible = []
                for a in remaining:
                    if all(p in scheduled for p in preds[a]):
                        if R > 0 and demands[a].size:
                            if np.all(demands[a] <= capacities - resource_usage + 1e-9):
                                eligible.append(a)
                        else:
                            eligible.append(a)

                if not eligible:
                    break

                # Use tail-based priority with some randomness
                if len(eligible) > 1 and random.random() < 0.3:
                    # Random selection for exploration
                    best_act = random.choice(eligible)
                else:
                    # Highest tail
                    best_act = max(eligible, key=lambda a: tail[a])

                start_time = current_time
                finish_time = start_time + durations[best_act]

                if R > 0 and demands[best_act].size:
                    resource_usage += demands[best_act]

                finish_times[best_act] = finish_time
                scheduled.add(best_act)
                remaining.discard(best_act)
                makespan = max(makespan, finish_time)
                current_time = finish_time

            makespan_sums[act] += makespan
            makespan_counts[act] += 1

    # Compute expected makespan for each eligible activity (lower is better)
    # For non-eligible activities, use tail as proxy
    expected_ms = np.zeros(n, dtype=float)
    for i in range(n):
        if i in eligible_set and makespan_counts[i] > 0:
            expected_ms[i] = makespan_sums[i] / makespan_counts[i]
        else:
            # Use tail as proxy for non-eligible
            expected_ms[i] = tail[i] + durations[i]

    # Normalize: lower expected makespan -> higher priority
    ms_max = expected_ms.max() if expected_ms.size else 1.0
    if ms_max > 0:
        ms_norm = expected_ms / ms_max
    else:
        ms_norm = expected_ms

    # Invert: higher priority for lower makespan
    ms_priority = 1.0 - ms_norm

    # Combine with base heuristics for non-eligible activities
    t_max = tail.max() if tail.size else 1.0
    if t_max > 0:
        tail_norm = tail / t_max
    else:
        tail_norm = tail

    sc_max = succ_count.max() if succ_count.size else 1.0
    if sc_max > 0:
        succ_norm = succ_count / sc_max
    else:
        succ_norm = succ_count

    d_max = durations.max() if durations.size else 1.0
    if d_max > 0:
        dur_norm = durations / d_max
    else:
        dur_norm = durations

    if R > 0 and capacities.size:
        cap = np.where(capacities <= 0, 1e-9, capacities)
        cap_max = cap.max() if cap.size else 1.0
        if cap_max > 0:
            cap_norm = cap / cap_max
        else:
            cap_norm = cap
        pressure_norm = np.sum(demands / cap_norm, axis=1)
        p_max = pressure_norm.max() if pressure_norm.size else 1.0
        if p_max > 0:
            pressure_norm = pressure_norm / p_max
    else:
        pressure_norm = np.zeros(n, dtype=float)

    # Final priority: for eligible, MCTS estimate dominates; for others, use heuristics
    prio = np.zeros(n, dtype=float)
    for i in range(n):
        if i in eligible_set:
            prio[i] = 1.0 * ms_priority[i] + 0.3 * tail_norm[i] + 0.1 * pressure_norm[i]
        else:
            prio[i] = 1.0 * tail_norm[i] + 0.3 * pressure_norm[i] + 0.15 * dur_norm[i] + 0.05 * succ_norm[i]

    if _elapsed() > _DEADLINE - _MARGIN:
        return np.asarray(durations, dtype=float)

    return np.asarray(prio, dtype=float)
