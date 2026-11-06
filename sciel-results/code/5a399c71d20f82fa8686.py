"""Rewritten program for RCPSP priority scoring using MCTS-inspired lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import heapq
import random

def _build_graph(n, predecessors):
    """Build successors and topological order."""
    successors = [set() for _ in range(n)]
    for i in range(n):
        for j in predecessors[i]:
            successors[j].add(i)
    
    in_degree = [0] * n
    for i in range(n):
        for j in predecessors[i]:
            in_degree[i] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for succ in successors[node]:
            temp_in_degree[succ] -= 1
            if temp_in_degree[succ] == 0:
                queue.append(succ)
    
    return successors, topo_order

def _compute_critical_paths(n, durations, predecessors, successors, topo_order):
    """Compute longest path from start to each node and from each node to end."""
    longest_from_start = [0.0] * n
    for node in topo_order:
        max_pred = 0.0
        for pred in predecessors[node]:
            max_pred = max(max_pred, longest_from_start[pred])
        longest_from_start[node] = max_pred + durations[node]
    
    longest_to_end = [0.0] * n
    for node in reversed(topo_order):
        max_succ = 0.0
        for succ in successors[node]:
            max_succ = max(max_succ, longest_to_end[succ])
        longest_to_end[node] = durations[node] + max_succ
    
    return longest_from_start, longest_to_end

def _simulate_schedule(instance, priority_fn, max_time=10000.0):
    """
    Serial schedule generation with resource checking.
    Returns makespan or None if timeout.
    """
    n = len(instance["durations"])
    if n == 0:
        return 0.0
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    if predecessors is None:
        predecessors = [set() for _ in range(n)]
    else:
        preds = []
        for p in predecessors:
            if p is None:
                preds.append(set())
            else:
                preds.append(set(p))
        predecessors = preds
    
    # Resource setup
    n_resources = 0
    if demands is not None and capacities is not None:
        n_resources = len(capacities)
        cap_vector = np.array(capacities, dtype=float)
        demand_matrix = np.zeros((n, n_resources), dtype=float)
        for i in range(n):
            if isinstance(demands[i], (list, tuple, np.ndarray)):
                for r in range(n_resources):
                    demand_matrix[i, r] = float(demands[i][r])
            else:
                demand_matrix[i, 0] = float(demands[i])
    else:
        n_resources = 0
        cap_vector = None
        demand_matrix = None
    
    # State
    scheduled = [False] * n
    finish_times = [0.0] * n
    start_times = [0.0] * n
    remaining = set(range(n))
    
    current_time = 0.0
    active_jobs = []  # list of (finish_time, job_id)
    
    def resource_usage_at(time):
        """Return resource usage vector at given time."""
        if n_resources == 0:
            return np.zeros(0)
        usage = np.zeros(n_resources)
        for ft, jid in active_jobs:
            if start_times[jid] <= time < ft:
                usage += demand_matrix[jid]
        return usage
    
    step_count = 0
    while remaining:
        step_count += 1
        if step_count > 10000:
            return None
        
        # Find eligible jobs
        eligible = []
        for j in remaining:
            all_done = True
            for k in predecessors[j]:
                if not scheduled[k]:
                    all_done = False
                    break
            if all_done:
                eligible.append(j)
        
        if not eligible:
            if not active_jobs:
                break
            # Advance to next finish time
            min_finish = min(ft for ft, _ in active_jobs)
            current_time = min_finish
            still_active = []
            for ft, jid in active_jobs:
                if ft == current_time:
                    scheduled[jid] = True
                    finish_times[jid] = current_time
                    remaining.discard(jid)
                else:
                    still_active.append((ft, jid))
            active_jobs = still_active
            continue
        
        # Try to schedule highest priority eligible job that fits
        eligible_sorted = sorted(eligible, key=lambda j: priority_fn[j], reverse=True)
        
        placed = False
        for j in eligible_sorted:
            if n_resources == 0:
                placed = True
                break
            
            # Check resource availability
            current_usage = resource_usage_at(current_time)
            new_usage = current_usage + demand_matrix[j]
            if np.all(new_usage <= cap_vector + 1e-9):
                placed = True
                break
        
        if not placed:
            if not active_jobs:
                # Deadlock? Force schedule highest priority.
                j = eligible_sorted[0]
                placed = True
            else:
                min_finish = min(ft for ft, _ in active_jobs)
                current_time = min_finish
                still_active = []
                for ft, jid in active_jobs:
                    if ft == current_time:
                        scheduled[jid] = True
                        finish_times[jid] = current_time
                        remaining.discard(jid)
                    else:
                        still_active.append((ft, jid))
                active_jobs = still_active
                continue
        
        # Schedule job j
        start_times[j] = current_time
        finish_times[j] = current_time + durations[j]
        scheduled[j] = True
        remaining.discard(j)
        active_jobs.append((finish_times[j], j))
    
    # Process remaining active
    while active_jobs:
        min_finish = min(ft for ft, _ in active_jobs)
        current_time = min_finish
        still_active = []
        for ft, jid in active_jobs:
            if ft == current_time:
                scheduled[jid] = True
                finish_times[jid] = current_time
            else:
                still_active.append((ft, jid))
        active_jobs = still_active
    
    makespan = max(finish_times) if n > 0 else 0.0
    return makespan

def priority(instance):
    """
    Score each activity using MCTS-inspired lookahead.
    For each activity, estimate the expected makespan if we prioritize it,
    then score it inversely to expected makespan (higher score = better priority).
    """
    n = len(instance["durations"])
    if n == 0:
        return np.array([], dtype=float)
    
    durations = np.asarray(instance["durations"], dtype=float)
    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    predecessors = instance.get("predecessors", None)
    
    if predecessors is None:
        predecessors = [set() for _ in range(n)]
    else:
        preds = []
        for p in predecessors:
            if p is None:
                preds.append(set())
            else:
                preds.append(set(p))
        predecessors = preds
    
    # Build graph
    successors, topo_order = _build_graph(n, predecessors)
    
    # If not all nodes in topo_order, there's a cycle; handle gracefully
    if len(topo_order) < n:
        return durations.copy()
    
    # Compute critical paths
    longest_from_start, longest_to_end = _compute_critical_paths(
        n, durations, predecessors, successors, topo_order
    )
    
    # Base priority: longest path to end (critical path)
    base_priority = np.array(longest_to_end, dtype=float)
    
    # Compute resource contention
    if demands is not None and capacities is not None:
        n_resources = len(capacities)
        cap_vector = np.array(capacities, dtype=float)
        demand_matrix = np.zeros((n, n_resources), dtype=float)
        for i in range(n):
            if isinstance(demands[i], (list, tuple, np.ndarray)):
                for r in range(n_resources):
                    demand_matrix[i, r] = float(demands[i][r])
            else:
                demand_matrix[i, 0] = float(demands[i])
        
        # Resource scarcity
        resource_scarcity = np.zeros(n_resources)
        for r in range(n_resources):
            users = [i for i in range(n) if demand_matrix[i, r] > 0]
            if users:
                total_demand = sum(demand_matrix[i, r] for i in users)
                scarcity = total_demand / max(cap_vector[r], 1e-9)
                resource_scarcity[r] = scarcity
        
        # Adjust priority based on resource scarcity
        scarcity_adjustment = np.zeros(n, dtype=float)
        for i in range(n):
            for r in range(n_resources):
                if demand_matrix[i, r] > 0:
                    scarcity_adjustment[i] += demand_matrix[i, r] * resource_scarcity[r]
        
        base_priority = base_priority + 0.5 * scarcity_adjustment
    
    # MCTS-inspired lookahead:
    # For each activity, create a priority vector that boosts it,
    # run a serial schedule, and record the makespan.
    # Then score[j] = -makespan_j (so higher score = lower makespan = better)
    
    # To keep it fast, we limit the number of simulations
    # Use a simple heuristic: boost the candidate activity and use
    # base_priority as tie-breaker
    
    scores = np.zeros(n, dtype=float)
    
    # For each activity, simulate with that activity boosted
    # To keep runtime manageable, we only do this for a subset or use a simpler model
    # We'll do a limited number of simulations per activity
    
    num_simulations = 5
    for j in range(n):
        makespans = []
        for _ in range(num_simulations):
            # Create a priority vector that boosts activity j
            priority_fn = base_priority.copy()
            # Boost activity j significantly
            priority_fn[j] = priority_fn[j] * 2.0 + 100.0
            
            # Run simulation
            makespan = _simulate_schedule(instance, priority_fn)
            if makespan is not None:
                makespans.append(makespan)
        
        if makespans:
            # Score is negative of average makespan (higher = better)
            scores[j] = -np.mean(makespans)
        else:
            # Fallback to base priority
            scores[j] = base_priority[j]
    
    # Normalize scores
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    return scores
# EVOLVE-BLOCK-END
