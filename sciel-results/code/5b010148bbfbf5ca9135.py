"""RCPSP priority via local search with optimized SGS simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict


def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy: 
    1. Compute features (Slack, Duration, Resource Criticality).
    2. Initialize priority based on Minimum Slack (strong heuristic).
    3. Use Hill Climbing / Simulated Annealing to refine priority scores
       by swapping or perturbing scores, evaluated via a fast SGS simulation.
    """
    start_time = time.time()
    DEADLINE = 118.0

    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

    # --- Extract instance components ---
    preds = None
    for key in ["predecessors", "preds", "predecessors_list", "pre", "pred"]:
        if key in instance:
            preds = instance[key]
            break
    if preds is None:
        preds = [[] for _ in range(n)]

    succs = None
    for key in ["successors", "succs", "successors_list", "succ"]:
        if key in instance:
            succs = instance[key]
            break

    demands = None
    for key in ["demands", "resource_demands", "resources", "dem", "resource"]:
        if key in instance:
            demands = instance[key]
            break
    if demands is None:
        demands = np.zeros((n, 1))

    capacities = None
    for key in ["capacities", "capacity", "resource_capacities", "caps", "cap"]:
        if key in instance:
            capacities = instance[key]
            break
    if capacities is None:
        capacities = [1.0]

    demands = np.asarray(demands, dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(n, 1)
    R = demands.shape[1]
    capacities = np.asarray(capacities, dtype=float)
    if capacities.ndim == 0:
        capacities = np.full(R, float(capacities))

    # Build successor lists if not provided
    if succs is None:
        succs = [set() for _ in range(n)]
        for j in range(n):
            for i in (preds[j] if isinstance(preds[j], (list, tuple, set, np.ndarray)) else []):
                idx = int(i)
                if 0 <= idx < n:
                    succs[idx].add(j)
        succs = [list(s) for s in succs]

    # --- Topological order ---
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        for i in (preds[j] if isinstance(preds[j], (list, tuple, set, np.ndarray)) else []):
            idx = int(i)
            if 0 <= idx < n:
                in_degree[j] += 1

    # Use a simple list as queue for topo sort
    q = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.pop(0)
        topo_order.append(u)
        for v in succs[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)

    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)

    # --- Forward pass ---
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        for v in succs[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]
        EF[u] = ES[u] + durations[u]

    makespan_forward = float(np.max(EF)) if n > 0 else 0.0

    # --- Backward pass ---
    LS = np.zeros(n)
    LF = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs[u]) == 0:
            LF[u] = makespan_forward
        else:
            LF[u] = float(np.min([LS[v] for v in succs[u]]))
        LS[u] = LF[u] - durations[u]
        for v in succs[u]:
            if LS[u] < LF[v]:
                LF[v] = LS[u]

    slack = LS - ES

    # --- Feature computation ---
    # Normalize slack to [0,1]
    min_slack = np.min(slack)
    max_slack = np.max(slack)
    if max_slack > min_slack:
        norm_slack = (slack - min_slack) / (max_slack - min_slack)
    else:
        norm_slack = np.zeros(n)
    
    # Priority: Higher is better. Low slack = High priority.
    # Base priority: 1 - norm_slack
    base_priority = 1.0 - norm_slack

    # --- Fast SGS Simulator (NumPy Optimized) ---
    def simulate_sgs(prio_scores):
        """
        Optimized Serial SGS using NumPy for resource checks.
        """
        if time.time() - start_time > DEADLINE:
            return float('inf')

        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        scheduled = np.zeros(n, dtype=bool)
        
        pred_list = [list(p) for p in preds]
        
        remaining = list(range(n))
        sched_indices = [] # List of indices of scheduled activities
        
        while remaining:
            if time.time() - start_time > DEADLINE:
                return float('inf')

            # Find eligible
            eligible = []
            for j in remaining:
                is_eligible = True
                for p in pred_list[j]:
                    if not scheduled[p]:
                        is_eligible = False
                        break
                if is_eligible:
                    eligible.append(j)

            if not eligible:
                for j in remaining:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                    scheduled[j] = True
                break

            # Pick highest priority
            best = max(eligible, key=lambda x: prio_scores[x])

            # Earliest start from precedence
            es_best = 0.0
            for p in pred_list[best]:
                cand = finish_t[p]
                if cand > es_best:
                    es_best = cand

            # Candidate times
            cand_times = {es_best}
            for k in sched_indices:
                if finish_t[k] >= es_best:
                    cand_times.add(finish_t[k])
            
            cand_times = sorted(cand_times)
            
            best_start = es_best
            found = False
            
            # Pre-allocate arrays for scheduled activities for faster NumPy ops
            # We only need to check scheduled activities
            if sched_indices:
                s_starts = start_t[sched_indices]
                s_ends = finish_t[sched_indices]
                s_demands = demands[sched_indices]
            else:
                s_starts = np.array([])
                s_ends = np.array([])
                s_demands = np.zeros((0, R))

            for t in cand_times:
                end_t = t + durations[best]
                
                # Check feasibility
                feasible = True
                
                # Identify events in [t, end_t)
                # We need to check load at all "breakpoints"
                # Breakpoints are t, end_t, and starts/ends of scheduled activities within (t, end_t)
                
                # To avoid set overhead, we can just check the load at:
                # 1. t
                # 2. Start times of any scheduled activity k where t <= start_k < end_t
                # Note: We don't strictly need to check end times for the *peak* if we just
                # evaluate the load function at the start points. The load drops at end times,
                # so the peak is always at a start time (or t).
                
                # So, check points: t, and start_k for all k in sched_indices if t <= start_k < end_t
                
                check_points = [t]
                for idx in range(len(s_starts)):
                    if t <= s_starts[idx] < end_t:
                        check_points.append(s_starts[idx])
                
                # Sort and unique (t is unique, starts are unique? No, multiple can start same time)
                # But we just need to check the load at these points.
                # Actually, if multiple start at same time, we check once.
                check_points = sorted(set(check_points))
                
                for tp in check_points:
                    # Calculate load at tp
                    # Active if start_k <= tp < end_k
                    # Note: We are checking interval [tp, tp+epsilon).
                    # So condition is start_k <= tp and end_k > tp.
                    
                    if len(s_starts) > 0:
                        mask = (s_starts <= tp) & (s_ends > tp)
                        current_load = s_demands[mask].sum(axis=0)
                    else:
                        current_load = np.zeros(R)
                    
                    # Add best activity
                    current_load += demands[best]
                    
                    if np.any(current_load > capacities + 1e-9):
                        feasible = False
                        break
                
                if feasible:
                    best_start = t
                    found = True
                    break

            if not found:
                best_start = es_best

            start_t[best] = best_start
            finish_t[best] = best_start + durations[best]
            scheduled[best] = True
            sched_indices.append(best)
            remaining.remove(best)

        makespan = float(np.max(finish_t)) if n > 0 else 0.0
        return makespan

    # --- Search Strategy ---
    # Start with Min Slack
    best_scores = base_priority.copy()
    best_ms = simulate_sgs(best_scores)
    
    # Multi-start Local Search
    # We will run several local search runs from different initializations
    
    max_time_per_run = (DEADLINE - (time.time() - start_time)) / 10.0 # 10 runs
    
    for run in range(10):
        if time.time() - start_time > DEADLINE - 1.0:
            break
        
        # Initialize
        if run == 0:
            current_scores = best_scores.copy()
        else:
            # Random initialization or perturbed best
            if random.random() < 0.5:
                current_scores = np.random.rand(n)
            else:
                current_scores = best_scores + np.random.randn(n) * 0.1
        
        current_ms = simulate_sgs(current_scores)
        
        # Local Search Loop
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() - start_time < DEADLINE - 1.0 and no_improve_count < max_no_improve:
            # Generate neighbor: Swap two random activities
            i1, i2 = random.sample(range(n), 2)
            neighbor_scores = current_scores.copy()
            neighbor_scores[i1], neighbor_scores[i2] = current_scores[i2], current_scores[i1]
            
            neighbor_ms = simulate_sgs(neighbor_scores)
            
            if neighbor_ms < current_ms:
                current_scores = neighbor_scores
                current_ms = neighbor_ms
                no_improve_count = 0
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_scores = current_scores.copy()
            else:
                no_improve_count += 1
                
    return best_scores
# EVOLVE-BLOCK-END
