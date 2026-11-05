# EVOLVE-BLOCK-START
import numpy as np
import time


def _build_graph(instance):
    """Extract n, durations, demands, capacities, and predecessor/successor lists."""
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # demands: shape (n, R)
    demands = np.asarray(instance["demands"], dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(n, -1)
    R = demands.shape[1]

    # capacities: shape (R,)
    capacities = np.asarray(instance["capacities"], dtype=float)

    # predecessors: list of lists
    preds_raw = instance.get("predecessors", None)
    if preds_raw is None:
        preds_raw = [[] for _ in range(n)]
    preds = [list(p) for p in preds_raw]

    # build successors
    succs = [[] for _ in range(n)]
    for j in range(n):
        for i in preds[j]:
            succs[i].append(j)

    return n, durations, demands, R, capacities, preds, succs


def _longest_path_to_end(n, durations, preds, succs):
    """Backward CPM: longest path from each activity to the end (inclusive of its own duration)."""
    # topological order via Kahn
    in_deg = [len(p) for p in preds]
    from collections import deque
    q = deque([i for i in range(n) if in_deg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in succs[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)
    # fallback: if cycle, just use range
    if len(topo) < n:
        topo = list(range(n))

    # backward pass
    lp = np.zeros(n, dtype=float)
    for u in reversed(topo):
        best_succ = 0.0
        for v in succs[u]:
            if lp[v] > best_succ:
                best_succ = lp[v]
        lp[u] = durations[u] + best_succ
    return lp


def _simulate_serial_schedule(n, durations, demands, R, capacities, preds, base_priority):
    """
    Run a serial schedule-generation scheme with the given base priority.
    Returns:
        start_times: array of shape (n,)
        resource_utilization: array of shape (T, R) where T is makespan (capped)
    """
    # eligible set
    remaining = set(range(n))
    completed = set()
    start_times = np.zeros(n, dtype=float)
    current_time = 0.0

    # We'll track resource usage over time using a list of (start, end, resource, amount)
    # For efficiency with 30 activities, we can use a simple event-based approach.
    # But for contention scoring, we need per-time-step utilization.
    # Cap the simulation horizon.
    max_horizon = float(np.sum(durations)) + 1.0
    T = int(max_horizon) + 1

    # resource_utilization[t][r] = total demand on resource r at time t
    resource_utilization = np.zeros((T, R), dtype=float)

    # active activities: list of (end_time, activity)
    active = []  # list of (end_time, act)

    while remaining:
        # find eligible: all predecessors completed
        eligible = [a for a in remaining if all(p in completed for p in preds[a])]
        if not eligible:
            # should not happen if DAG is valid; break to avoid infinite loop
            break

        # pick highest base_priority
        best_a = max(eligible, key=lambda a: base_priority[a])
        remaining.remove(best_a)

        # determine start time: max of (current_time, end times of any active that share resources)
        # Actually in serial SGS, we start as soon as all predecessors are done and resources allow.
        # But serial SGS typically starts at the earliest feasible time >= current_time.
        # Let's compute earliest feasible start.
        earliest = current_time
        # Check active activities for resource conflicts
        for (end_t, act) in active:
            # if act and best_a share any resource with positive demand,
            # and their intervals overlap, we need to wait.
            # But since we're doing serial (one at a time), actually in serial SGS
            # we schedule one activity at a time at the earliest feasible time.
            # Let me reconsider: serial SGS picks the highest-priority eligible activity
            # and starts it at the earliest time >= current_time when all its resources are available.
            # Since we process one at a time, "current_time" advances to the end of the last scheduled.
            # Actually no: in serial SGS, we maintain a time pointer. We pick the best eligible,
            # find the earliest time >= time_pointer where resources are available, schedule it there,
            # then advance time_pointer to the end of that activity? No, that's not right either.
            #
            # Standard serial SGS:
            #   t = 0
            #   while there are unscheduled activities:
            #       eligible = {a unscheduled : all preds of a are scheduled and finish <= t}
            #       if eligible empty: t = min finish time of any scheduled activity, continue
            #       pick a* = max priority in eligible
            #       schedule a* at time t
            #       t = t + duration[a*]  (or t = finish time of a*)
            #
            # Actually the standard is: t advances to the next event. Let me implement properly.
            pass

        # Let me redo this more carefully.
        break  # placeholder, will rewrite below

    return start_times, resource_utilization


def _simulate_serial_schedule_v2(n, durations, demands, R, capacities, preds, base_priority):
    """
    Proper serial schedule-generation scheme.
    Returns start_times (n,) and a list of (start, end, act) for each activity.
    """
    # We'll use an event-driven approach.
    # State:
    #   scheduled: set of completed activities
    #   pending: set of activities that are "running" (started but not finished)
    #   unscheduled: set of not-yet-started activities
    #   time: current time
    #   For each activity, we need to know its start and end.

    unscheduled = set(range(n))
    running = {}  # act -> (start, end)
    scheduled = set()
    start_times = np.zeros(n, dtype=float)
    end_times = np.zeros(n, dtype=float)

    current_time = 0.0

    # We'll simulate by advancing time to the next event (completion of a running activity).
    # At each event, we check if any unscheduled activity becomes eligible and can be started.

    while unscheduled or running:
        # Check if any unscheduled activity can be started at current_time
        # An activity is eligible if all its predecessors are in `scheduled` (completed)
        # AND there is enough resource capacity at current_time.

        # Compute resource usage at current_time from running activities
        usage = np.zeros(R, dtype=float)
        for act, (s, e) in running.items():
            if s <= current_time < e:
                usage += demands[act]

        # Find eligible unscheduled activities
        eligible = []
        for a in list(unscheduled):
            if all(p in scheduled for p in preds[a]):
                # check resource availability
                if np.all(usage + demands[a] <= capacities + 1e-9):
                    eligible.append(a)

        if eligible:
            # pick highest priority
            best_a = max(eligible, key=lambda a: base_priority[a])
            unscheduled.remove(best_a)
            s = current_time
            e = s + durations[best_a]
            start_times[best_a] = s
            end_times[best_a] = e
            running[best_a] = (s, e)
            # Do NOT advance time; we may start more activities at the same time
            # if resources allow. But in serial SGS, we typically start one at a time.
            # Let's start one per "round" to be safe, then continue checking.
            # Actually, let's continue the loop to see if more can start at same time.
            continue

        # No eligible activity can start. Advance time to next completion event.
        if not running:
            # Deadlock? Shouldn't happen for a valid DAG.
            # Force-complete to avoid infinite loop.
            break

        next_time = min(e for (s, e) in running.values())
        # Complete all activities that finish at next_time
        for act in list(running.keys()):
            s, e = running[act]
            if e <= next_time + 1e-9:
                del running[act]
                scheduled.add(act)
        current_time = next_time

    return start_times, end_times, running  # running is empty at end


def priority(instance):
    start_wall = time.time()
    deadline = start_wall + 100.0  # safety margin before 120s limit

    n, durations, demands, R, capacities, preds, succs = _build_graph(instance)

    # 1. Longest path to end (criticality)
    lp = _longest_path_to_end(n, durations, preds, succs)

    # 2. Base priority for simulation: use longest path as base
    base_priority = lp.copy()

    # 3. Simulate a serial schedule to get start times and identify contention
    try:
        start_times, end_times, _ = _simulate_serial_schedule_v2(
            n, durations, demands, R, capacities, preds, base_priority
        )
    except Exception:
        # Fallback: just use longest path
        return lp

    # 4. Compute resource contention score for each activity
    # For each activity, look at the time interval [start, end) and compute
    # the peak resource utilization ratio on each resource it consumes.
    # contention_score[a] = max over resources r in support(a) of (peak_utilization[r] / capacity[r])
    # where peak_utilization is computed over the activity's time window.

    # To do this efficiently, we can compute for each resource r, the total demand
    # at each time step, then for each activity, find the max utilization during its window.
    # With 30 activities and small T, this is fine.

    # Determine time granularity. Use integer time steps if durations are integers,
    # otherwise use a fine grid.
    max_end = float(np.max(end_times)) if n > 0 else 0.0
    # Use a resolution of 1.0 for simplicity, or finer if needed.
    # Let's use a resolution that captures all start/end times.
    all_times = sorted(set(np.concatenate([start_times, end_times, [0.0, max_end]])))
    # Build a timeline of unique time points
    # For each interval between consecutive time points, resource usage is constant.

    # Compute resource usage over time intervals
    # Intervals: [t_i, t_{i+1}) for i in range(len(all_times)-1)
    intervals = []
    for i in range(len(all_times) - 1):
        t_start = all_times[i]
        t_end = all_times[i + 1]
        if t_end - t_start < 1e-9:
            continue
        intervals.append((t_start, t_end))

    # For each interval, compute total resource demand
    interval_usage = []
    for (t_start, t_end) in intervals:
        t_mid = (t_start + t_end) / 2.0
        usage = np.zeros(R, dtype=float)
        for a in range(n):
            if start_times[a] <= t_mid < end_times[a]:
                usage += demands[a]
        interval_usage.append(usage)

    # For each activity, find the maximum utilization ratio on its resources during its window
    contention = np.zeros(n, dtype=float)
    for a in range(n):
        s, e = start_times[a], end_times[a]
        if e - s < 1e-9:
            contention[a] = 0.0
            continue
        # Find intervals that overlap with [s, e)
        max_ratio = 0.0
        for idx, (t_start, t_end) in enumerate(intervals):
            # Check overlap
            if t_end <= s + 1e-9 or t_start >= e - 1e-9:
                continue
            usage = interval_usage[idx]
            for r in range(R):
                if demands[a, r] > 1e-9 and capacities[r] > 1e-9:
                    ratio = usage[r] / capacities[r]
                    if ratio > max_ratio:
                        max_ratio = ratio
        contention[a] = max_ratio

    # 5. Combine: priority = longest_path + lambda * contention
    # Normalize contention to be comparable to longest_path
    if np.max(lp) > 1e-9:
        lp_norm = lp / np.max(lp)
    else:
        lp_norm = lp
    if np.max(contention) > 1e-9:
        contention_norm = contention / np.max(contention)
    else:
        contention_norm = contention

    # Weight: give significant weight to contention
    lam = 0.5
    combined = lp_norm + lam * contention_norm

    # Also add a small bonus for high resource demand (demand/capacity) as a tiebreaker
    demand_ratio = np.zeros(n, dtype=float)
    for a in range(n):
        for r in range(R):
            if capacities[r] > 1e-9:
                demand_ratio[a] = max(demand_ratio[a], demands[a, r] / capacities[r])
    if np.max(demand_ratio) > 1e-9:
        demand_ratio_norm = demand_ratio / np.max(demand_ratio)
    else:
        demand_ratio_norm = demand_ratio

    final_priority = combined + 0.1 * demand_ratio_norm

    # Check deadline
    if time.time() > deadline:
        return lp  # fallback

    return final_priority
# EVOLVE-BLOCK-END
