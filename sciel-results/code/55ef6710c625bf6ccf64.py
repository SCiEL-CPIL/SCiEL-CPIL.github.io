"""RCPSP priority via fast event-based SGS with coordinate descent optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict
import heapq


def priority(instance):
    """
    Compute a priority score for each activity in a Resource-Constrained
    Project Scheduling Problem (RCPSP) instance.
    
    Strategy: Fast event-based SGS + coordinate descent optimization.
    """
    start_time = time.time()
    DEADLINE = 117.0  # safety margin before 120s kill

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
        features[i, 0] = 1.0 / s
        features[i, 1] = durations[i]
        features[i, 2] = np.sum(demands[i])
        features[i, 3] = float(len(succs[i]))
        features[i, 4] = float(len(pred_set[i]))
        features[i, 5] = 1.0 / (d * s)

    # --- Fast Event-Based SGS Simulator ---
    def simulate_sgs(prio_scores):
        """
        Serial SGS with event-based resource tracking.
        O(n * log(n) * R) per simulation.
        """
        start_t = np.full(n, np.inf)
        finish_t = np.full(n, np.inf)
        
        # Track resource usage via events: (time, resource_delta, activity_id)
        # We'll maintain current resource usage and use a heap for next events
        
        remaining = set(range(n))
        unscheduled_preds = [len(pred_set[j]) for j in range(n)]
        
        # Event heap: (time, type, activity_id)
        # type 0 = start, type 1 = finish
        event_heap = []
        current_usage = np.zeros(R)
        active_activities = set()
        
        scheduled = 0
        
        while scheduled < n:
            # Check for deadline
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
            # We need to find the first time t >= es_best where resources are sufficient
            
            # Get all candidate times: es_best and finish times of scheduled activities >= es_best
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
                # An activity k is active at time t if start_t[k] < t < finish_t[k]
                # Or if start_t[k] == t (starting now)
                
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

    # --- Optimize weights using coordinate descent ---
    n_features = features.shape[1]
    
    # Initialize with reasonable defaults
    best_weights = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.5])
    best_scores = features @ best_weights
    best_ms = simulate_sgs(best_scores)
    
    current_weights = best_weights.copy()
    current_ms = best_ms
    
    # Coordinate descent with random restarts
    max_rounds = 50
    steps_per_dim = 10
    
    for round_num in range(max_rounds):
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        improved = False
        for d in range(n_features):
            if time.time() - start_time > DEADLINE - 2.0:
                break
            
            # Try moving weight d in both directions
            for direction in [-1, 1]:
                step_size = 0.1
                for step in range(steps_per_dim):
                    neighbor_weights = current_weights.copy()
                    neighbor_weights[d] += direction * step_size
                    
                    # Clamp
                    neighbor_weights = np.clip(neighbor_weights, -3.0, 3.0)
                    
                    neighbor_scores = features @ neighbor_weights
                    neighbor_ms = simulate_sgs(neighbor_scores)
                    
                    if neighbor_ms < current_ms - 1e-9:
                        current_weights = neighbor_weights
                        current_ms = neighbor_ms
                        improved = True
                        if current_ms < best_ms:
                            best_ms = current_ms
                            best_weights = current_weights.copy()
                    else:
                        break
        
        if not improved:
            # Random restart: perturb all weights
            if time.time() - start_time > DEADLINE - 2.0:
                break
            current_weights = best_weights + np.random.randn(n_features) * 0.3
            current_weights = np.clip(current_weights, -3.0, 3.0)
            current_ms = simulate_sgs(features @ current_weights)
            if current_ms < best_ms:
                best_ms = current_ms
                best_weights = current_weights.copy()
    
    # Final priority scores
    best_scores = features @ best_weights
    return best_scores
# EVOLVE-BLOCK-END
