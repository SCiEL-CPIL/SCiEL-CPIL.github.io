"""
RCPSP Priority Function
Computes priority scores for activities to minimize makespan in a serial SGS.
Uses Best-of-Heuristics with strong RCPSP heuristics as baseline,
then refines via targeted local search on the schedule.
"""
import numpy as np
import time
from collections import defaultdict, deque

def priority(instance):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin (5s buffer before 120s limit)

    # 1. Extract Data
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    preds_list = instance.get("predecessors", instance.get("preds", [[] for _ in range(n)]))
    succs_list = instance.get("successors", instance.get("succs", None))
    
    demands = instance.get("demands", instance.get("resource_demands", None))
    capacities = instance.get("capacities", instance.get("resource_capacities", None))

    # Normalize demands and capacities
    if demands is None:
        demands = np.zeros((n, 1))
    demands = np.asarray(demands, dtype=float)
    if demands.ndim == 1:
        demands = demands.reshape(n, -1)
        
    R = demands.shape[1]
    
    if capacities is None:
        capacities = np.ones(R)
    capacities = np.asarray(capacities, dtype=float)
    if capacities.ndim == 0:
        capacities = np.full(R, float(capacities))

    # Build successor lists if not provided
    if succs_list is None:
        succs_list = [[] for _ in range(n)]
        for j in range(n):
            for p in preds_list[j]:
                succs_list[int(p)].append(j)
    else:
        succs_list = [list(map(int, s)) for s in succs_list]
        preds_list = [list(map(int, p)) for p in preds_list]

    # 2. Topological Sort
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        in_degree[j] = len(preds_list[j])
        
    q = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v in succs_list[u]:
            in_deg_copy[v] -= 1
            if in_deg_copy[v] == 0:
                q.append(v)
                
    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)

    # 3. CPM Analysis
    ES = np.zeros(n)
    EF = np.zeros(n)
    for u in topo_order:
        for v in succs_list[u]:
            if EF[u] > ES[v]:
                ES[v] = EF[u]
        EF[u] = ES[u] + durations[u]
        
    makespan_unconstrained = float(np.max(EF)) if n > 0 else 0.0
    
    LS = np.zeros(n)
    LF = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs_list[u]) == 0:
            LF[u] = makespan_unconstrained
        else:
            LF[u] = float(np.min([LS[v] for v in succs_list[u]]))
        LS[u] = LF[u] - durations[u]
        for v in succs_list[u]:
            if LS[u] < LF[v]:
                LF[v] = LS[u]
                
    slack = LS - ES
    
    # Critical Rank: Longest path from i to end
    rank = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs_list[u]) == 0:
            rank[u] = durations[u]
        else:
            rank[u] = durations[u] + float(np.max([rank[v] for v in succs_list[u]]))

    # Resource Scarcity
    if R > 0:
        safe_caps = np.where(capacities == 0, 1.0, capacities)
        resource_intensity = np.sum(demands / safe_caps, axis=1)
    else:
        resource_intensity = np.zeros(n)

    # 4. Fast SGS Simulator with optimized resource checking
    def simulate_sgs(scores, pred_sets=None, succ_sets=None):
        """
        Simulates Serial SGS with given priority scores.
        Returns makespan and schedule (start, finish times).
        """
        if time.time() - start_time > DEADLINE:
            return float('inf'), None, None
            
        if pred_sets is None:
            pred_sets = [set(p) for p in preds_list]
        if succ_sets is None:
            succ_sets = [set(s) for s in succs_list]
            
        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        remaining = set(range(n))
        
        # Track which activities are scheduled for faster eligibility check
        scheduled_count = np.zeros(n, dtype=int)  # count of scheduled predecessors
        
        while remaining:
            if time.time() - start_time > DEADLINE:
                return float('inf'), None, None
                
            # Find eligible activities
            eligible = []
            for j in remaining:
                if all(start_t[p] < np.inf for p in pred_sets[j]):
                    eligible.append(j)
            
            if not eligible:
                # Fallback
                for j in remaining:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                break
                
            # Pick highest priority
            best = max(eligible, key=lambda x: scores[x])
            
            # Precedence constraint earliest start
            es_prec = 0.0
            for p in pred_sets[best]:
                if start_t[p] + durations[p] > es_prec:
                    es_prec = start_t[p] + durations[p]
            
            # Find earliest feasible time considering resources
            candidate_times = set()
            candidate_times.add(es_prec)
            for k in range(n):
                if finish_t[k] < np.inf:
                    candidate_times.add(finish_t[k])
                    if start_t[k] >= es_prec:
                        candidate_times.add(start_t[k])
            
            candidates = sorted([t for t in candidate_times if t >= es_prec])
            
            if not candidates:
                candidates = [es_prec]
                
            best_start = None
            for t in candidates:
                end_t = t + durations[best]
                feasible = True
                
                # Check resource constraints
                check_pts = set([t, end_t])
                for k in range(n):
                    if start_t[k] < np.inf:
                        if t <= start_t[k] < end_t:
                            check_pts.add(start_t[k])
                        if t < finish_t[k] <= end_t:
                            check_pts.add(finish_t[k])
                
                check_pts = sorted(check_pts)
                
                for i in range(len(check_pts) - 1):
                    tp1 = check_pts[i]
                    tp2 = check_pts[i+1]
                    if tp2 <= t or tp1 >= end_t:
                        continue
                    
                    # Sum demands for this interval
                    active_demands = np.zeros(R)
                    active_demands += demands[best]
                    for k in range(n):
                        if start_t[k] < np.inf:
                            if start_t[k] <= tp1 and finish_t[k] >= tp2:
                                active_demands += demands[k]
                    
                    if np.any(active_demands > capacities + 1e-9):
                        feasible = False
                        break
                
                if feasible:
                    best_start = t
                    break
            
            if best_start is None:
                best_start = es_prec
                
            start_t[best] = best_start
            finish_t[best] = best_start + durations[best]
            remaining.discard(best)
            
            # Update scheduled_count for successors
            for s in succ_sets[best]:
                scheduled_count[s] += 1
            
        if n == 0:
            return 0.0, start_t, finish_t
        return float(np.max(finish_t)), start_t, finish_t

    # 5. Define strong heuristic priority rules
    # Each rule returns a priority score vector (higher = more urgent)
    
    def rule_min_slack():
        """Minimum Total Slack: schedule activities with least slack first"""
        s_min, s_max = np.min(slack), np.max(slack)
        if s_max > s_min:
            return (s_max - slack) / (s_max - s_min)
        return np.zeros(n)
    
    def rule_max_rank():
        """Maximum Critical Path Rank: schedule activities with longest remaining path first"""
        r_min, r_max = np.min(rank), np.max(rank)
        if r_max > r_min:
            return (rank - r_min) / (r_max - r_min)
        return np.zeros(n)
    
    def rule_min_slack_plus_rank():
        """Combination of Min Slack and Max Rank"""
        s = rule_min_slack()
        r = rule_max_rank()
        return 0.5 * s + 0.5 * r
    
    def rule_resource_critical():
        """Resource Criticality: prioritize activities using scarce resources"""
        if R == 0:
            return np.zeros(n)
        r_min, r_max = np.min(resource_intensity), np.max(resource_intensity)
        if r_max > r_min:
            return (resource_intensity - r_min) / (r_max - r_min)
        return np.zeros(n)
    
    def rule_duration_desc():
        """Longest Processing Time: schedule longer activities first"""
        d_min, d_max = np.min(durations), np.max(durations)
        if d_max > d_min:
            return (durations - d_min) / (d_max - d_min)
        return np.zeros(n)
    
    def rule_successor_count():
        """Most Successors: schedule activities that unlock most work first"""
        succ_count = np.array([len(s) for s in succs_list], dtype=float)
        sc_min, sc_max = np.min(succ_count), np.max(succ_count)
        if sc_max > sc_min:
            return (succ_count - sc_min) / (sc_max - sc_min)
        return np.zeros(n)
    
    def rule_combined_strong():
        """Strong combination: 0.4*MinSlack + 0.3*MaxRank + 0.2*ResCritical + 0.1*Duration"""
        s = rule_min_slack()
        r = rule_max_rank()
        rc = rule_resource_critical()
        d = rule_duration_desc()
        return 0.4 * s + 0.3 * r + 0.2 * rc + 0.1 * d
    
    # List of heuristics to try
    heuristics = [
        ("min_slack", rule_min_slack),
        ("max_rank", rule_max_rank),
        ("min_slack_plus_rank", rule_min_slack_plus_rank),
        ("resource_critical", rule_resource_critical),
        ("duration_desc", rule_duration_desc),
        ("successor_count", rule_successor_count),
        ("combined_strong", rule_combined_strong),
    ]
    
    # 6. Best-of-Heuristics: evaluate each heuristic
    pred_sets = [set(p) for p in preds_list]
    succ_sets = [set(s) for s in succs_list]
    
    best_ms = float('inf')
    best_scores = None
    best_start_t = None
    best_finish_t = None
    
    for name, rule_func in heuristics:
        if time.time() - start_time > DEADLINE - 30:
            break
        scores = rule_func()
        # Add tiny noise to break ties deterministically
        scores = scores + np.arange(n) * 1e-10
        ms, st, ft = simulate_sgs(scores, pred_sets, succ_sets)
        if ms < best_ms:
            best_ms = ms
            best_scores = scores.copy()
            best_start_t = st
            best_finish_t = ft
    
    # 7. Local Search: try swapping priority of pairs of activities
    # Focus on activities that are close in priority or on the critical path
    if time.time() - start_time < DEADLINE - 15:
        # Identify activities to consider for swaps: those with high resource usage or on critical paths
        # For simplicity, consider all pairs but limit the number of swaps
        max_swaps = 50
        swaps_done = 0
        
        # Create a copy of scores for local search
        ls_scores = best_scores.copy()
        
        # Try local improvements: for each pair (i, j), if swapping their relative priority improves makespan
        # To be efficient, only consider pairs where both are not fixed by precedence
        for _ in range(max_swaps):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Pick a random pair to swap
            i, j = np.random.choice(n, 2, replace=False)
            
            # Skip if i must precede j or vice versa (swapping wouldn't help much)
            if j in pred_sets[i] or i in pred_sets[j]:
                continue
            
            # Swap their scores
            old_i, old_j = ls_scores[i], ls_scores[j]
            ls_scores[i], ls_scores[j] = old_j, old_i
            
            ms_new, st_new, ft_new = simulate_sgs(ls_scores, pred_sets, succ_sets)
            
            if ms_new < best_ms - 1e-9:
                best_ms = ms_new
                best_scores = ls_scores.copy()
                best_start_t = st_new
                best_finish_t = ft_new
            else:
                # Revert swap
                ls_scores[i], ls_scores[j] = old_i, old_j
                ls_scores = best_scores.copy()
            
            swaps_done += 1
    
    # 8. Final refinement: try small perturbations around best solution
    if time.time() - start_time < DEADLINE - 5:
        for _ in range(20):
            if time.time() - start_time > DEADLINE - 2:
                break
            
            perturbed = best_scores.copy()
            # Add small random perturbation
            idx = np.random.choice(n, 3, replace=False)
            for idx_i in idx:
                perturbed[idx_i] += np.random.uniform(-0.1, 0.1)
            
            ms_new, st_new, ft_new = simulate_sgs(perturbed, pred_sets, succ_sets)
            
            if ms_new < best_ms - 1e-9:
                best_ms = ms_new
                best_scores = perturbed.copy()
                best_start_t = st_new
                best_finish_t = ft_new
    
    # Return the best scores found
    if best_scores is None:
        # Fallback to combined strong heuristic
        best_scores = rule_combined_strong()
    
    # Ensure scores are distinct by adding tiny index-based offset
    final_scores = best_scores + np.arange(n) * 1e-12
    
    return final_scores
