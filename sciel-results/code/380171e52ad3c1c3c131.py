"""Improved program for RCPSP using look-ahead simulation."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time


def priority(instance):
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    if n == 0:
        return np.array([], dtype=float)

    # Get predecessors with defensive key handling
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred", "precedence"]:
        if key in instance:
            preds = instance[key]
            break

    if preds is None:
        # No precedence constraints: return duration-based priority (longer first)
        return np.asarray(durations, dtype=float)

    # Normalize preds to list of lists of ints
    if isinstance(preds, np.ndarray):
        preds = preds.tolist()

    # Build successor list
    succs = [set() for _ in range(n)]
    for j in range(n):
        if j < len(preds):
            p_list = preds[j]
            if p_list is None:
                continue
            for i in p_list:
                if i is not None:
                    i = int(i)
                    if 0 <= i < n and i != j:
                        succs[i].add(j)

    # Check deadline
    if time.time() - start_time > DEADLINE:
        return np.asarray(durations, dtype=float)

    # Get resources
    demands = None
    capacities = None
    for key in ["demands", "resource_demands", "demand", "resource_demand"]:
        if key in instance:
            demands = instance[key]
            break
    for key in ["capacities", "resource_capacities", "capacity", "resource_capacity"]:
        if key in instance:
            capacities = instance[key]
            break

    demands = np.asarray(demands, dtype=float) if demands is not None else np.zeros((n, 1))
    capacities = np.asarray(capacities, dtype=float) if capacities is not None else np.ones(1)
    if demands.ndim == 1:
        demands = demands.reshape(-1, 1)
    if capacities.ndim == 0:
        capacities = capacities.reshape(1)
    n_resources = capacities.shape[0]

    # Precompute in-degrees for quick eligibility checks
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in succs[j]:
            in_degree[i] += 1

    def simulate_serial_schedule(initial_choice=None, max_time=0.5):
        """
        Simulate serial schedule generation.
        If initial_choice is given, force that activity to be scheduled first.
        Otherwise, use a greedy heuristic to pick the next eligible activity.
        Returns the makespan.
        """
        sim_start = time.time()
        scheduled = set()
        remaining_indeg = in_degree.copy()
        current_time = 0.0
        resource_usage = np.zeros(n_resources)
        event_list = []  # (completion_time, activity)
        
        # If we have an initial choice, schedule it first
        if initial_choice is not None and initial_choice not in scheduled:
            # Check if it's eligible (all preds scheduled)
            eligible = True
            pred_list = preds[initial_choice] if initial_choice < len(preds) and preds[initial_choice] else []
            for p in pred_list:
                if p is not None and int(p) not in scheduled:
                    eligible = False
                    break
            if eligible:
                scheduled.add(initial_choice)
                end_time = current_time + durations[initial_choice]
                event_list.append((end_time, initial_choice))
                resource_usage += demands[initial_choice]
                for s in succs[initial_choice]:
                    remaining_indeg[s] -= 1
                current_time = end_time

        while len(scheduled) < n:
            if time.time() - sim_start > max_time:
                return float('inf')
            
            # Process all events that complete at or before current_time
            progressed = True
            while progressed:
                progressed = False
                new_events = []
                for (et, act) in event_list:
                    if et <= current_time:
                        resource_usage -= demands[act]
                        for s in succs[act]:
                            remaining_indeg[s] -= 1
                        progressed = True
                    else:
                        new_events.append((et, act))
                event_list = new_events
                if not progressed:
                    break
            
            # Find eligible activities: all predecessors scheduled, and resource feasible
            eligible_activities = []
            for j in range(n):
                if j in scheduled:
                    continue
                pred_list = preds[j] if j < len(preds) and preds[j] else []
                all_preds_done = True
                for p in pred_list:
                    if p is not None and int(p) not in scheduled:
                        all_preds_done = False
                        break
                if not all_preds_done:
                    continue
                feasible = True
                for r in range(n_resources):
                    if resource_usage[r] + demands[j, r] > capacities[r]:
                        feasible = False
                        break
                if feasible:
                    eligible_activities.append(j)
            
            if not eligible_activities:
                if event_list:
                    current_time = min(et for et, _ in event_list)
                    continue
                else:
                    return float('inf')
            
            # Pick the eligible activity with highest priority using a simple heuristic
            best_act = None
            best_score = -float('inf')
            for j in eligible_activities:
                score = len(succs[j]) * 0.5 + durations[j] * 0.3 + np.sum(demands[j] / np.where(capacities > 0, capacities, 1.0)) * 0.2
                if score > best_score:
                    best_score = score
                    best_act = j
            
            if best_act is None:
                best_act = eligible_activities[0]
            
            scheduled.add(best_act)
            end_time = current_time + durations[best_act]
            event_list.append((end_time, best_act))
            resource_usage += demands[best_act]
            for s in succs[best_act]:
                remaining_indeg[s] -= 1
            
            any_can_start_now = False
            for j in range(n):
                if j in scheduled:
                    continue
                pred_list = preds[j] if j < len(preds) and preds[j] else []
                all_preds_done = True
                for p in pred_list:
                    if p is not None and int(p) not in scheduled:
                        all_preds_done = False
                        break
                if not all_preds_done:
                    continue
                feasible = True
                for r in range(n_resources):
                    if resource_usage[r] + demands[j, r] > capacities[r]:
                        feasible = False
                        break
                if feasible:
                    any_can_start_now = True
                    break
            
            if not any_can_start_now:
                if event_list:
                    current_time = min(et for et, _ in event_list)
                else:
                    current_time = end_time
        
        if event_list:
            makespan = max(et for et, _ in event_list)
        else:
            makespan = current_time
        return makespan

    # Identify initially eligible activities
    initially_eligible = []
    for j in range(n):
        pred_list = preds[j] if j < len(preds) and preds[j] else []
        all_preds_done = True
        for p in pred_list:
            if p is not None:
                all_preds_done = False
                break
        if all_preds_done:
            feasible = True
            for r in range(n_resources):
                if demands[j, r] > capacities[r]:
                    feasible = False
                    break
            if feasible:
                initially_eligible.append(j)

    # Get baseline makespan
    baseline_makespan = simulate_serial_schedule(initial_choice=None, max_time=0.3)
    
    if time.time() - start_time > DEADLINE * 0.7:
        return _static_priority(durations, preds, succs, demands, capacities, n)

    priority_scores = np.zeros(n, dtype=float)
    
    if initially_eligible:
        for j in initially_eligible:
            if time.time() - start_time > DEADLINE * 0.8:
                break
            makespan_j = simulate_serial_schedule(initial_choice=j, max_time=0.3)
            if np.isinf(makespan_j):
                priority_scores[j] = 0.0
            else:
                priority_scores[j] = 1.0 / (1.0 + makespan_j)
        
        topo_dist = np.full(n, -1, dtype=int)
        queue = deque(initially_eligible)
        for j in initially_eligible:
            topo_dist[j] = 0
        while queue:
            u = queue.popleft()
            for v in succs[u]:
                if topo_dist[v] < 0:
                    topo_dist[v] = topo_dist[u] + 1
                    queue.append(v)
        
        max_dist = np.max(topo_dist[topo_dist >= 0]) if np.any(topo_dist >= 0) else 1
        for j in range(n):
            if j not in initially_eligible:
                if topo_dist[j] >= 0:
                    priority_scores[j] = 0.3 * (1.0 - topo_dist[j] / max(max_dist, 1))
                else:
                    priority_scores[j] = 0.1
    else:
        return _static_priority(durations, preds, succs, demands, capacities, n)

    if time.time() - start_time > DEADLINE:
        return _static_priority(durations, preds, succs, demands, capacities, n)

    return priority_scores


def _static_priority(durations, preds, succs, demands, capacities, n):
    """Fallback static heuristic priority."""
    if n == 0:
        return np.array([], dtype=float)
    
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in succs[j]:
            in_degree[i] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                queue.append(v)
    
    if len(topo_order) < n:
        remaining = [i for i in range(n) if i not in set(topo_order)]
        topo_order.extend(remaining)
    
    ES = np.zeros(n, dtype=float)
    for u in topo_order:
        for v in succs[u]:
            if ES[u] + durations[u] > ES[v]:
                ES[v] = ES[u] + durations[u]
    
    project_duration = np.max(ES + durations) if n > 0 else 0
    
    LS = np.full(n, project_duration, dtype=float)
    for u in reversed(topo_order):
        for v in succs[u]:
            if LS[v] - durations[u] < LS[u]:
                LS[u] = LS[v] - durations[u]
    
    total_float = LS - ES
    
    if demands is None or capacities is None:
        scarcity = np.zeros(n, dtype=float)
    else:
        safe_cap = np.where(capacities > 0, capacities, 1.0)
        scarcity = np.sum(demands / safe_cap, axis=1)
    
    succ_count = np.array([len(s) for s in succs], dtype=float)
    
    def normalize(x):
        if len(x) == 0:
            return x
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    tf_norm = normalize(-total_float)
    sc_norm = normalize(scarcity)
    succ_norm = normalize(succ_count)
    dur_norm = normalize(durations)
    
    return (0.50 * tf_norm + 0.25 * sc_norm + 0.15 * succ_norm + 0.10 * dur_norm)
# EVOLVE-BLOCK-END
