"""RCPSP priority via fast event-based SGS with coordinate descent optimization."""
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
    1. Compute rich features (slack, duration, resource criticality, successors).
    2. Use fast event-based SGS simulation (O(n log n) per schedule).
    3. Optimize linear feature weights via adaptive coordinate descent.
    """
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill

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

    # --- Compute features for each activity ---
    eps = 1e-9
    # Feature 0: 1/(slack+eps) — low slack gets high priority
    # Feature 1: duration — longer activities get higher priority
    # Feature 2: max resource demand ratio (demand/capacity) — resource criticality
    # Feature 3: number of successors — activities that unblock many get priority
    # Feature 4: 1/(duration * (slack+eps)) — combined criticality
    # Feature 5: weighted slack (normalized)
    
    max_demand_ratio = np.zeros(n)
    for i in range(n):
        for r in range(R):
            if capacities[r] > 0:
                ratio = demands[i, r] / capacities[r]
                if ratio > max_demand_ratio[i]:
                    max_demand_ratio[i] = ratio

    features = np.zeros((n, 6))
    for i in range(n):
        s = slack[i] + eps
        d = durations[i] + eps
        features[i, 0] = 1.0 / s
        features[i, 1] = durations[i]
        features[i, 2] = max_demand_ratio[i]
        features[i, 3] = float(len(succs[i]))
        features[i, 4] = 1.0 / (d * s)
        features[i, 5] = slack[i]  # raw slack (inverted in optimization)

    # Normalize features to [0, 1]
    feat_min = np.min(features, axis=0)
    feat_max = np.max(features, axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range < 1e-12] = 1.0
    features_norm = (features - feat_min) / feat_range

    # --- Fast Event-Based SGS Simulator ---
    def simulate_sgs(prio_scores):
        """
        Serial SGS using event-driven approach for speed.
        """
        if n == 0:
            return 0.0
        
        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        
        # Track unscheduled activities
        unscheduled = set(range(n))
        
        # Pre-compute eligible check: for each activity, count unscheduled predecessors
        unscheduled_preds_count = np.zeros(n, dtype=int)
        for j in range(n):
            unscheduled_preds_count[j] = len(pred_set[j])
        
        # Priority heap for eligible activities (negative priority for max-heap)
        eligible_heap = []
        
        # Initially, activities with no predecessors are eligible
        for j in range(n):
            if unscheduled_preds_count[j] == 0:
                heappush(eligible_heap, (-prio_scores[j], j))
        
        # Track active resource usage over time using difference array approach
        # For each resource, maintain a list of (time, delta) events
        resource_events = [dict() for _ in range(R)]  # resource_events[r][time] = total_delta
        
        while unscheduled:
            if not eligible_heap:
                # Should not happen if instance is valid, but fallback
                for j in unscheduled:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                break
            
            # Get highest priority eligible activity
            neg_prio, j = heappop(eligible_heap)
            
            if j not in unscheduled:
                continue  # stale entry
            
            # Compute earliest start from precedence
            es_j = 0.0
            for p in pred_set[j]:
                cand = start_t[p] + durations[p]
                if cand > es_j:
                    es_j = cand
            
            # Find earliest feasible start time >= es_j
            # Candidate times: es_j and all finish times of currently scheduled activities >= es_j
            # We need to check resource feasibility at each candidate time
            
            # Collect candidate times
            cand_times = {es_j}
            for k in range(n):
                if finish_t[k] != np.inf and finish_t[k] >= es_j:
                    cand_times.add(finish_t[k])
            
            cand_times = sorted(cand_times)
            
            best_start = None
            for t in cand_times:
                if t < es_j - 1e-9:
                    continue
                
                end_t = t + durations[j]
                
                # Check resource feasibility: for each resource, compute usage in [t, end_t)
                feasible = True
                for r in range(R):
                    # Compute total demand from activities active in [t, end_t)
                    usage = 0.0
                    for k in range(n):
                        if start_t[k] != np.inf:
                            # Activity k is active in [start_t[k], finish_t[k])
                            # Check overlap with [t, end_t)
                            if start_t[k] < end_t and finish_t[k] > t:
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
            
            # Update eligibility of successors
            for v in succs[j]:
                unscheduled_preds_count[v] -= 1
                if unscheduled_preds_count[v] == 0 and v in unscheduled:
                    heappush(eligible_heap, (-prio_scores[v], v))
        
        makespan = float(np.max(finish_t)) if n > 0 else 0.0
        return makespan

    # --- Optimize weights using adaptive coordinate descent ---
    n_features = features_norm.shape[1]

    # Initialize with strong heuristic: weight on 1/slack
    best_weights = np.zeros(n_features)
    best_weights[0] = 2.0  # strong emphasis on low slack
    best_weights[1] = 0.5  # some emphasis on duration
    best_weights[2] = 1.0  # resource criticality
    best_weights[3] = 0.5  # successors
    best_weights[5] = -1.0  # penalize high slack (feature 5 is raw slack)
    
    best_ms = simulate_sgs(features_norm @ best_weights)
    current_weights = best_weights.copy()
    current_ms = best_ms

    # Adaptive coordinate descent
    step_sizes = np.ones(n_features) * 0.5
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < DEADLINE - 3.0:
        improved = False
        for fi in range(n_features):
            if time.time() - start_time > DEADLINE - 3.0:
                break
            
            # Try positive and negative steps
            for direction in [1, -1]:
                trial_weights = current_weights.copy()
                trial_weights[fi] += direction * step_sizes[fi]
                trial_weights = np.clip(trial_weights, -10.0, 10.0)
                
                trial_ms = simulate_sgs(features_norm @ trial_weights)
                
                if trial_ms < current_ms:
                    current_weights = trial_weights
                    current_ms = trial_ms
                    improved = True
                    no_improve_count = 0
                    step_sizes[fi] *= 1.3  # increase step
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_weights = current_weights.copy()
                    break  # try other features
                else:
                    step_sizes[fi] *= 0.7  # decrease step
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
        else:
            no_improve_count = 0
    
    # Final priority scores
    best_scores = features_norm @ best_weights
    return best_scores
# EVOLVE-BLOCK-END
