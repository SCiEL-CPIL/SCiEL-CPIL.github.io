"""RCPSP priority via Fast SGS and Nelder-Mead Simplex Optimization on Normalized Features."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy:
    1. Compute heuristic features (slack, duration, resource demand, etc.).
    2. Normalize features to [0, 1] to stabilize optimization.
    3. Use a fast Serial SGS simulator.
    4. Optimize linear combination weights using Nelder-Mead Simplex method.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s kill

    # Set a seed for reproducibility
    random.seed(42)
    np.random.seed(42)

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

    # --- Compute features for each activity ---
    eps = 1e-9
    features = np.zeros((n, 6))
    for i in range(n):
        s = slack[i] + eps
        d = durations[i] + eps
        features[i, 0] = 1.0 / s          # Inverse slack (criticality)
        features[i, 1] = durations[i]      # Duration
        features[i, 2] = np.sum(demands[i])# Total resource demand
        features[i, 3] = float(len(succs[i])) # Successors count
        features[i, 4] = float(len(pred_set[i])) # Predecessors count
        features[i, 5] = 1.0 / (d * s)     # Inverse duration*slack

    # Normalize features to [0, 1] for stable optimization
    feat_min = np.min(features, axis=0)
    feat_max = np.max(features, axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range < 1e-12] = 1.0
    features_norm = (features - feat_min) / feat_range

    # --- Fast SGS Simulator ---
    def simulate_sgs(prio_scores):
        """
        Serial SGS with event-based resource tracking.
        Optimized with active set tracking.
        """
        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        
        remaining = set(range(n))
        unscheduled_preds = [len(pred_set[j]) for j in range(n)]
        
        scheduled = 0
        # We don't strictly need active_at_time dict if we scan, but scanning is O(n^2) per step.
        # For n=30, O(n^2) is 900 ops per step, 30 steps -> 27000 ops. Very fast.
        # However, checking resources for every candidate time is the bottleneck.
        
        while scheduled < n:
            if time.time() - start_time > DEADLINE - 1.0:
                break
            
            # Find eligible activities (all preds scheduled)
            eligible = []
            for j in remaining:
                if unscheduled_preds[j] == 0:
                    eligible.append(j)
            
            if not eligible:
                # Fallback: schedule remaining at ES
                for j in remaining:
                    start_t[j] = ES[j]
                    finish_t[j] = ES[j] + durations[j]
                break
            
            # Pick highest priority eligible
            best = max(eligible, key=lambda x: prio_scores[x])
            
            # Earliest start from precedence
            es_best = 0.0
            for p in pred_set[best]:
                cand = finish_t[p]
                if cand > es_best:
                    es_best = cand
            
            # Find earliest feasible start time >= es_best
            # Candidate times: es_best, and all future finish times
            cand_times = [es_best]
            for k in range(n):
                if finish_t[k] != np.inf and finish_t[k] >= es_best - 1e-9:
                    cand_times.append(finish_t[k])
            
            cand_times = sorted(set(cand_times))
            
            best_start = None
            for t in cand_times:
                if t < es_best - 1e-9:
                    continue
                
                # Check if resources are available at time t
                # Only consider activities that are active at time t
                active_at_t = np.zeros(R)
                for k in range(n):
                    if start_t[k] != np.inf and start_t[k] < t + 1e-9 and finish_t[k] > t - 1e-9:
                        active_at_t += demands[k]
                
                if np.all(active_at_t + demands[best] <= capacities + 1e-9):
                    best_start = t
                    break
            
            if best_start is None:
                # Shouldn't happen, but fallback
                best_start = es_best
            
            # Schedule activity
            start_t[best] = best_start
            finish_t[best] = best_start + durations[best]
            remaining.discard(best)
            scheduled += 1
            
            # Update unscheduled preds for successors
            for v in succs[best]:
                if v in remaining:
                    unscheduled_preds[v] -= 1
        
        makespan = float(np.max(finish_t)) if n > 0 and np.all(finish_t != np.inf) else float(np.max(finish_t[finish_t != np.inf])) if np.any(finish_t != np.inf) else 0.0
        return makespan

    # --- Nelder-Mead Simplex Optimization ---
    n_features = features_norm.shape[1]
    
    # Initialize simplex
    # Start with a few random points and standard basis perturbations
    base_w = np.ones(n_features) / n_features
    simplex = [base_w.copy()]
    
    # Add initial diverse points
    simplex.append(base_w + np.array([0.5, 0, 0, 0, 0, 0]))
    simplex.append(base_w + np.array([0, 0.5, 0, 0, 0, 0]))
    simplex.append(base_w + np.array([0, 0, 0.5, 0, 0, 0]))
    simplex.append(base_w + np.array([0, 0, 0, 0.5, 0, 0]))
    simplex.append(base_w + np.array([0, 0, 0, 0, 0.5, 0]))
    simplex.append(base_w + np.array([0, 0, 0, 0, 0, 0.5]))
    
    # Evaluate initial simplex
    values = [simulate_sgs(features_norm @ w) for w in simplex]
    
    best_idx = np.argmin(values)
    best_ms = values[best_idx]
    best_weights = simplex[best_idx].copy()
    
    alpha = 1.0   # Reflection
    gamma = 2.0   # Expansion
    rho = 0.5     # Contraction
    sigma = 0.5   # Shrink
    
    max_iter = 2000
    min_delta = 1e-6
    
    for it in range(max_iter):
        if time.time() - start_time > DEADLINE - 2.0:
            break
            
        # Sort simplex by value
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        if values[0] < best_ms:
            best_ms = values[0]
            best_weights = simplex[0].copy()
            
        # Check convergence
        if max(values) - min(values) < min_delta:
            break
            
        worst = simplex[-1]
        worst_val = values[-1]
        
        # Centroid of best n points
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        reflected = centroid - alpha * (worst - centroid)
        reflected_val = simulate_sgs(features_norm @ reflected)
        
        if reflected_val < values[0]:
            # Expansion
            expanded = centroid + gamma * (reflected - centroid)
            expanded_val = simulate_sgs(features_norm @ expanded)
            if expanded_val < reflected_val:
                simplex[-1] = expanded
                values[-1] = expanded_val
            else:
                simplex[-1] = reflected
                values[-1] = reflected_val
        elif reflected_val < values[-2]:
            # Accept reflection
            simplex[-1] = reflected
            values[-1] = reflected_val
        else:
            # Contraction
            if reflected_val < worst_val:
                # Outside contraction
                contracted = centroid + rho * (worst - centroid)
                contracted_val = simulate_sgs(features_norm @ contracted)
                if contracted_val <= worst_val:
                    simplex[-1] = contracted
                    values[-1] = contracted_val
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        values[i] = simulate_sgs(features_norm @ simplex[i])
            else:
                # Inside contraction
                contracted = centroid - rho * (worst - centroid)
                contracted_val = simulate_sgs(features_norm @ contracted)
                if contracted_val <= worst_val:
                    simplex[-1] = contracted
                    values[-1] = contracted_val
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        values[i] = simulate_sgs(features_norm @ simplex[i])
                        
    # Final priority scores
    best_scores = features_norm @ best_weights
    return best_scores
# EVOLVE-BLOCK-END
