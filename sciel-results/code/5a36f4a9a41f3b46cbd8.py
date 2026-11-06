"""RCPSP priority via non-linear multiplicative scoring with optimized SGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict, deque
from heapq import heappush, heappop


def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy:
    1. Compute slack, duration, resource criticality via forward/backward pass.
    2. Use a non-linear multiplicative priority: duration^alpha * (1/(slack+eps))^beta * resource_crit^gamma
    3. Optimize exponents (alpha, beta, gamma) via simulated annealing on makespan.
    4. Fast SGS with efficient resource feasibility via active-set tracking.
    """
    start_time = time.time()
    DEADLINE = 118.0  # safety margin before 120s kill

    # --- Extract instance components ---
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)

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

    # Build pred sets for fast lookup
    pred_set = [set() for _ in range(n)]
    for j in range(n):
        p = preds[j]
        if isinstance(p, (list, tuple, set, np.ndarray)):
            for i in p:
                idx = int(i)
                if 0 <= idx < n:
                    pred_set[j].add(idx)

    # --- Topological order ---
    in_degree = np.zeros(n, dtype=int)
    for j in range(n):
        in_degree[j] = len(pred_set[j])

    q = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    in_deg_copy = in_degree.copy()
    while q:
        u = q.popleft()
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

    slack = LS - ES

    # --- Compute base features for non-linear scoring ---
    eps = 1e-9
    max_demand_ratio = np.zeros(n)
    for i in range(n):
        for r in range(R):
            if capacities[r] > 0:
                ratio = demands[i, r] / capacities[r]
                if ratio > max_demand_ratio[i]:
                    max_demand_ratio[i] = ratio

    # Compute downstream count (number of activities that transitively depend on this one)
    # Use reverse topological order to accumulate counts
    downstream_count = np.zeros(n, dtype=float)
    for u in reversed(topo_order):
        # downstream_count[u] = 1 (itself) + sum of downstream_count of direct successors? No.
        # Actually, we want the number of nodes reachable from u.
        # Let's compute reachability counts via DP in reverse topo order.
        # reach[u] = 1 + sum(reach[v] for v in succs[u]) is not correct because of diamond structures.
        # We need the size of the transitive closure.
        # For small n (30), we can just do a BFS/DFS from each node.
        pass
    
    # Since n is small (30), let's compute exact downstream counts via BFS from each node
    downstream_count = np.zeros(n, dtype=float)
    for i in range(n):
        visited = set()
        stack = [i]
        while stack:
            node = stack.pop()
            for v in succs[node]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        downstream_count[i] = len(visited)
    
    # Normalize to avoid overflow/underflow in power operations
    dur_norm = durations / (np.max(durations) + eps)
    slack_norm = slack / (np.max(slack) + eps)
    crit_norm = max_demand_ratio / (np.max(max_demand_ratio) + eps)
    succ_count = np.array([len(succs[i]) for i in range(n)], dtype=float)
    succ_norm = succ_count / (np.max(succ_count) + eps)
    down_norm = downstream_count / (np.max(downstream_count) + eps)

    # --- Fast SGS with active-set resource tracking ---
    def simulate_sgs(prio_scores):
        """
        Serial SGS with efficient resource feasibility check.
        Uses sorted list of active activities for O(log n) resource queries.
        """
        if n == 0:
            return 0.0
        
        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        
        unscheduled = set(range(n))
        unscheduled_preds_count = np.array([len(pred_set[j]) for j in range(n)], dtype=int)
        
        eligible_heap = []
        for j in range(n):
            if unscheduled_preds_count[j] == 0:
                heappush(eligible_heap, (-prio_scores[j], j))
        
        # Track scheduled activities for resource checks
        scheduled = []  # list of (start, finish, activity_idx)
        
        while unscheduled:
            if not eligible_heap:
                for j in unscheduled:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                break
            
            neg_prio, j = heappop(eligible_heap)
            
            if j not in unscheduled:
                continue
            
            # Compute earliest start from precedence
            es_j = 0.0
            for p in pred_set[j]:
                cand = start_t[p] + durations[p]
                if cand > es_j:
                    es_j = cand
            
            # Candidate times: es_j and finish times of scheduled activities >= es_j
            cand_times = [es_j]
            for (s_k, f_k, k) in scheduled:
                if f_k >= es_j:
                    cand_times.append(f_k)
            cand_times = sorted(set(cand_times))
            
            best_start = None
            for t in cand_times:
                if t < es_j - 1e-9:
                    continue
                
                end_t = t + durations[j]
                
                # Check resource feasibility
                feasible = True
                for r in range(R):
                    usage = 0.0
                    for (s_k, f_k, k) in scheduled:
                        if s_k < end_t and f_k > t:
                            usage += demands[k, r]
                    
                    if usage + demands[j, r] > capacities[r] + 1e-9:
                        feasible = False
                        break
                
                if feasible:
                    best_start = t
                    break
            
            if best_start is None:
                best_start = es_j
            
            start_t[j] = best_start
            finish_t[j] = best_start + durations[j]
            unscheduled.discard(j)
            scheduled.append((best_start, finish_t[j], j))
            
            for v in succs[j]:
                unscheduled_preds_count[v] -= 1
                if unscheduled_preds_count[v] == 0 and v in unscheduled:
                    heappush(eligible_heap, (-prio_scores[v], v))
        
        makespan = float(np.max(finish_t)) if n > 0 else 0.0
        return makespan

    # --- Non-linear multiplicative priority function ---
    def compute_priority(alpha, beta, gamma, delta, epsilon):
        """
        Priority = duration^alpha * (1/(slack+eps))^beta * resource_crit^gamma * succ_count^delta * downstream^epsilon
        All components normalized to [0,1] range, so this is well-behaved.
        """
        # Avoid zero values causing issues
        dur_safe = np.maximum(dur_norm, eps)
        slack_inv_safe = np.maximum(1.0 - slack_norm, eps)  # 1/(slack+eps) approximated
        crit_safe = np.maximum(crit_norm, eps)
        succ_safe = np.maximum(succ_norm + eps, eps)
        down_safe = np.maximum(down_norm + eps, eps)
        
        prio = (dur_safe ** alpha) * (slack_inv_safe ** beta) * (crit_safe ** gamma) * (succ_safe ** delta) * (down_safe ** epsilon)
        return prio

    # --- Simulated Annealing optimization of exponents ---
    best_params = np.array([1.0, 2.0, 1.0, 0.5, 0.5])  # initial: alpha, beta, gamma, delta, epsilon
    best_ms = simulate_sgs(compute_priority(*best_params))
    current_params = best_params.copy()
    current_ms = best_ms
    
    T = 1.0  # temperature
    T_min = 0.01
    cooling_rate = 0.95
    max_iters = 500
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Generate neighbor
        new_params = current_params + rng.normal(0, 0.3, size=5)
        new_params = np.clip(new_params, 0.0, 5.0)
        
        new_ms = simulate_sgs(compute_priority(*new_params))
        
        # Accept if better or with probability exp(-delta/T)
        delta = new_ms - current_ms
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
            current_params = new_params
            current_ms = new_ms
            if current_ms < best_ms:
                best_ms = current_ms
                best_params = current_params.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final priority scores
    best_scores = compute_priority(*best_params)
    return best_scores
# EVOLVE-BLOCK-END
