"""
RCPSP Priority Function
Computes priority scores for activities to minimize makespan in a serial SGS.
Uses a combination of critical path length, slack, resource scarcity, and contention.
Performs a lightweight local search over feature weights to optimize the specific instance.
"""
import numpy as np
import time
from collections import defaultdict, deque

def priority(instance):
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin

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
    
    rank = np.zeros(n)
    for u in reversed(topo_order):
        if len(succs_list[u]) == 0:
            rank[u] = durations[u]
        else:
            rank[u] = durations[u] + float(np.max([rank[v] for v in succs_list[u]]))

    if R > 0:
        safe_caps = np.where(capacities == 0, 1.0, capacities)
        resource_intensity = np.sum(demands / safe_caps, axis=1)
    else:
        resource_intensity = np.zeros(n)

    # 4. Compute Resource Contention: sum of shared demand with immediate successors
    # High contention means this activity competes with successors for resources
    contention = np.zeros(n)
    if R > 0:
        for i in range(n):
            for s in succs_list[i]:
                # Sum of min(demand_i[r], demand_s[r]) / capacity[r] for each resource
                shared = np.minimum(demands[i], demands[s])
                contention[i] += np.sum(shared / safe_caps)
    # Normalize contention
    cont_min, cont_max = np.min(contention), np.max(contention)
    if cont_max > cont_min:
        feat_cont = (contention - cont_min) / (cont_max - cont_min)
    else:
        feat_cont = np.zeros(n)

    # Features
    # We want HIGH score = HIGH Priority (schedule early)
    
    # 1. Inverse Slack
    slack_min, slack_max = np.min(slack), np.max(slack)
    if slack_max > slack_min:
        norm_slack = (slack - slack_min) / (slack_max - slack_min)
    else:
        norm_slack = np.zeros(n)
    feat_slack = 1.0 - norm_slack

    # 2. Rank
    rank_min, rank_max = np.min(rank), np.max(rank)
    if rank_max > rank_min:
        feat_rank = (rank - rank_min) / (rank_max - rank_min)
    else:
        feat_rank = np.ones(n)

    # 3. Resource Intensity
    res_min, res_max = np.min(resource_intensity), np.max(resource_intensity)
    if res_max > res_min:
        feat_res = (resource_intensity - res_min) / (res_max - res_min)
    else:
        feat_res = np.zeros(n)

    # 4. Successor Count
    succ_count = np.array([len(s) for s in succs_list], dtype=float)
    sc_min, sc_max = np.min(succ_count), np.max(succ_count)
    if sc_max > sc_min:
        feat_succ = (succ_count - sc_min) / (sc_max - sc_min)
    else:
        feat_succ = np.zeros(n)

    # 5. Duration
    dur_min, dur_max = np.min(durations), np.max(durations)
    if dur_max > dur_min:
        feat_dur = (durations - dur_min) / (dur_max - dur_min)
    else:
        feat_dur = np.zeros(n)

    # 6. Contention
    # feat_cont already computed above

    # Feature Matrix (n x 6)
    features = np.column_stack([
        feat_slack, 
        feat_rank, 
        feat_res, 
        feat_succ, 
        feat_dur,
        feat_cont
    ])

    # 5. Fast SGS Simulator (optimized with event tracking)
    def simulate_sgs(scores):
        """
        Simulates Serial SGS with given priority scores.
        Returns makespan.
        """
        if time.time() - start_time > DEADLINE:
            return float('inf')
            
        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        
        pred_sets = [set(p) for p in preds_list]
        remaining = set(range(n))
        
        while remaining:
            if time.time() - start_time > DEADLINE:
                return float('inf')
                
            # Find eligible activities
            eligible = [j for j in remaining if all(start_t[p] < np.inf for p in pred_sets[j])]
            
            if not eligible:
                for j in remaining:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                break
                
            # Pick highest priority
            best = max(eligible, key=lambda x: scores[x])
            
            # Precedence constraint earliest start
            es_prec = 0.0
            for p in pred_sets[best]:
                ft = start_t[p] + durations[p]
                if ft > es_prec:
                    es_prec = ft
            
            # Find earliest feasible time considering resources
            candidate_times = {es_prec}
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
                
                # Critical points for resource check
                check_pts = {t, end_t}
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
                    
                    active_demands = demands[best].copy()
                    for k in range(n):
                        if start_t[k] < np.inf and start_t[k] <= tp1 and finish_t[k] >= tp2:
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
            
        if n == 0:
            return 0.0
        return float(np.max(finish_t))

    # 6. Weight Search with expanded candidate set
    n_feats = 6  # slack, rank, res, succ, dur, cont
    
    candidate_weights = [
        np.ones(n_feats) / n_feats,
        np.array([0.25, 0.25, 0.15, 0.10, 0.10, 0.15]),
        np.array([0.35, 0.35, 0.10, 0.05, 0.05, 0.10]),
        np.array([0.20, 0.40, 0.15, 0.05, 0.05, 0.15]),
        np.array([0.40, 0.30, 0.10, 0.05, 0.05, 0.10]),
        np.array([0.15, 0.20, 0.35, 0.05, 0.05, 0.20]),
        np.array([0.15, 0.25, 0.15, 0.10, 0.15, 0.20]),
        np.array([0.05, 0.50, 0.20, 0.05, 0.05, 0.15]),
        np.array([0.25, 0.25, 0.25, 0.05, 0.05, 0.15]),
        np.array([0.30, 0.30, 0.10, 0.05, 0.05, 0.20]),  # Slack+Rank+Cont
        np.array([0.10, 0.30, 0.25, 0.05, 0.05, 0.25]),  # Rank+Res+Cont
        np.array([0.20, 0.20, 0.20, 0.10, 0.10, 0.20]),  # Balanced with cont
    ]
    
    best_ms = float('inf')
    best_w = np.ones(n_feats) / n_feats
    
    for w in candidate_weights:
        if time.time() - start_time > DEADLINE - 5:
            break
        scores = features @ w
        scores = scores + np.random.uniform(0, 1e-6, n)
        ms = simulate_sgs(scores)
        if ms < best_ms:
            best_ms = ms
            best_w = w.copy()

    # Local Perturbation around best found - increased iterations
    for _ in range(25):
        if time.time() - start_time > DEADLINE - 2:
            break
        w_pert = best_w + np.random.normal(0, 0.05, n_feats)
        w_pert = np.clip(w_pert, 0, 1)
        s = np.sum(w_pert)
        if s > 0:
            w_pert = w_pert / s
        else:
            continue
            
        scores = features @ w_pert
        scores = scores + np.random.uniform(0, 1e-6, n)
        ms = simulate_sgs(scores)
        if ms < best_ms:
            best_ms = ms
            best_w = w_pert.copy()

    # Final Scores
    final_scores = features @ best_w
    final_scores = final_scores + np.random.uniform(0, 1e-6, n)
    
    return final_scores
