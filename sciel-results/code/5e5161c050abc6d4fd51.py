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

    # 2. Forward longest path (earliest start time without resource constraints)
    # Compute ES (earliest start) for each activity
    in_deg = [len(p) for p in preds]
    from collections import deque
    q = deque([i for i in range(n) if in_deg[i] == 0])
    es = np.zeros(n, dtype=float)
    while q:
        u = q.popleft()
        for v in succs[u]:
            es[v] = max(es[v], es[u] + durations[u])
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)

    # 3. Compute slack: total float = LF - LS = (lp_from_start + lp_to_end) - total_project_time
    # lp_from_start[u] = es[u], lp_to_end[u] = lp[u] (backward pass already computed)
    # Total project time (without resources) = max(es[u] + durations[u] for all u) = max(lp)
    project_duration_no_res = float(np.max(lp)) if n > 0 else 0.0
    slack = np.zeros(n, dtype=float)
    for u in range(n):
        slack[u] = max(0.0, project_duration_no_res - es[u] - lp[u])

    # 4. Base priority: combine criticality (low slack = high priority) with longest path
    # Use inverse slack as a proxy for criticality
    max_slack = float(np.max(slack)) if n > 0 else 0.0
    if max_slack > 1e-9:
        inv_slack = 1.0 / (1.0 + slack / max_slack)  # smooth inverse, range (0.5, 1.0]
    else:
        inv_slack = np.ones(n, dtype=float)

    # Normalize lp
    max_lp = float(np.max(lp)) if n > 0 else 1.0
    lp_norm = lp / max_lp if max_lp > 1e-9 else lp

    # Combined criticality score
    crit_score = 0.6 * lp_norm + 0.4 * inv_slack

    # 5. Simulate a serial schedule using criticality as base priority
    base_priority = crit_score.copy()
    try:
        start_times, end_times, _ = _simulate_serial_schedule_v2(
            n, durations, demands, R, capacities, preds, base_priority
        )
    except Exception:
        # Fallback: just use criticality score
        return crit_score

    # 6. Compute resource contention: how much each activity contributes to peak utilization
    # For each resource, find the time of peak utilization and identify which activities
    # are running at that peak. Activities contributing to peaks get higher contention scores.
    
    # Build timeline of unique event times
    all_times = sorted(set(np.concatenate([start_times, end_times, [0.0]])))
    if len(all_times) < 2:
        all_times.append(1.0)
    
    # Compute resource usage at each interval midpoint
    num_intervals = len(all_times) - 1
    interval_usage = np.zeros((num_intervals, R), dtype=float)
    for i in range(num_intervals):
        t_mid = (all_times[i] + all_times[i + 1]) / 2.0
        for a in range(n):
            if start_times[a] <= t_mid < end_times[a]:
                interval_usage[i] += demands[a]
    
    # For each resource, find peak utilization and which activities contribute
    contention = np.zeros(n, dtype=float)
    
    # Smooth max: use log-sum-exp for a soft version of max
    # For each activity, compute its "contribution to scarcity"
    # An activity's contention = sum over resources it uses of:
    #   (demand[a,r] / capacity[r]) * (peak_usage[r] / capacity[r])
    # This weights the demand by how scarce the resource is at peak.
    
    # First, compute peak usage per resource
    peak_usage = np.zeros(R, dtype=float)
    for r in range(R):
        if num_intervals > 0:
            peak_usage[r] = float(np.max(interval_usage[:, r]))
        else:
            peak_usage[r] = 0.0
    
    # Compute scarcity factor per resource: peak_usage / capacity, clipped to [0, 1]
    scarcity = np.zeros(R, dtype=float)
    for r in range(R):
        if capacities[r] > 1e-9:
            scarcity[r] = min(1.0, peak_usage[r] / capacities[r])
        else:
            scarcity[r] = 0.0
    
    # For each activity, contention = sum over r of demand[a,r] * scarcity[r] / capacity[r]
    # This measures how much the activity demands scarce resources
    for a in range(n):
        c = 0.0
        for r in range(R):
            if capacities[r] > 1e-9 and demands[a, r] > 1e-9:
                c += (demands[a, r] / capacities[r]) * scarcity[r]
        contention[a] = c
    
    # Also compute "blocking" score: how many other activities are waiting for resources
    # that this activity is using. Approximate by: for each time step where this activity
    # is running, count how many other eligible-but-blocked activities exist.
    # Simpler proxy: duration * max_scarcity_of_its_resources
    max_scarcity_per_act = np.zeros(n, dtype=float)
    for a in range(n):
        for r in range(R):
            if demands[a, r] > 1e-9:
                max_scarcity_per_act[a] = max(max_scarcity_per_act[a], scarcity[r])
    blocking = durations * max_scarcity_per_act
    
    # 7. Combine all components
    # Normalize contention and blocking
    max_cont = float(np.max(contention)) if n > 0 else 1.0
    max_block = float(np.max(blocking)) if n > 0 else 1.0
    cont_norm = contention / max_cont if max_cont > 1e-9 else contention
    block_norm = blocking / max_block if max_block > 1e-9 else blocking
    
    # Final priority: criticality dominates, with resource scarcity as secondary
    # The insight: prioritize critical activities that also consume scarce resources
    final_priority = crit_score + 0.35 * cont_norm + 0.15 * block_norm
    
    # Check deadline
    if time.time() > deadline:
        return crit_score  # fallback

    return final_priority
# EVOLVE-BLOCK-END
