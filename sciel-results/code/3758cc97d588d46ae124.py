"""Combined program: Structural Critical Path Priority + Best-Fit Packing Refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import deque

def _compute_features(times, preds, cycle_time):
    """Compute structural features for priority scoring."""
    n = len(times)
    if n == 0:
        return {}
    
    # Precompute successors
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
    
    # Topological sort
    in_degree = [len(preds[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for s in succ[node]:
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    
    # Handle potential cycles or disconnected nodes by appending remaining
    if len(topo_order) < n:
        for i in range(n):
            if i not in topo_order:
                topo_order.append(i)

    # Bottom-level rank (Longest path to end, including own time)
    bottom_level = np.zeros(n, dtype=float)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        max_succ = 0.0
        for s in succ[node]:
            if bottom_level[s] > max_succ:
                max_succ = bottom_level[s]
        bottom_level[node] = times[node] + max_succ
        
    # Total downstream workload (Sum of times of all successors + own time)
    total_downstream = np.zeros(n, dtype=float)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        total_downstream[node] = times[node]
        for s in succ[node]:
            total_downstream[node] += total_downstream[s]
            
    # Immediate successor count (Fan-out)
    num_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Critical Ratio: Downstream Workload / Own Time
    # High ratio indicates a task that unlocks a lot of work relative to its own cost.
    safe_times = np.where(times == 0, 1e-9, times)
    critical_ratio = total_downstream / safe_times
    
    features = {
        'bottom_level': bottom_level,
        'total_downstream': total_downstream,
        'num_succ': num_succ,
        'critical_ratio': critical_ratio,
        'succ': succ,
        'topo_order': topo_order
    }
    
    return features

def _normalize(arr):
    """Normalize array to [0, 1] range."""
    mn = np.min(arr)
    mx = np.max(arr)
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def _greedy_assign_best_fit(times, preds, cycle_time, priority_scores):
    """
    Simulate greedy station filling.
    Improvement: Among eligible tasks that fit, select the one that minimizes 
    the remaining space (Best Fit). If there's a tie in remaining space, 
    select the one with the highest priority score.
    """
    n = len(times)
    if n == 0:
        return 0

    # Precompute successors for fast eligibility updates
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)

    remaining_preds = [len(preds[i]) for i in range(n)]
    eligible_list = [i for i in range(n) if remaining_preds[i] == 0]
    
    stations = 0
    remaining = n
    
    while remaining > 0:
        stations += 1
        load = 0.0
        
        while True:
            best_task = -1
            best_remaining_space = float('inf')
            best_prio = -np.inf
            
            # Find the best eligible task that fits
            # Strategy: Best Fit (minimize waste) with Priority tie-breaking
            for i in eligible_list:
                if times[i] <= cycle_time - load + 1e-9:
                    remaining_space = cycle_time - load - times[i]
                    prio = priority_scores[i]
                    
                    # We prefer tasks that fill the station most tightly.
                    # If remaining_space is smaller, it's a better fit.
                    # If remaining_space is equal, we prefer higher priority.
                    if (remaining_space < best_remaining_space - 1e-9) or \
                       (abs(remaining_space - best_remaining_space) < 1e-9 and prio > best_prio):
                        best_remaining_space = remaining_space
                        best_prio = prio
                        best_task = i
            
            if best_task == -1:
                break
            
            # Assign best_task
            load += times[best_task]
            remaining -= 1
            eligible_list.remove(best_task)
            
            # Update successors
            for s in succ[best_task]:
                remaining_preds[s] -= 1
                if remaining_preds[s] == 0:
                    eligible_list.append(s)
    
    return stations

def priority(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    if n == 0:
        return np.array([])
    
    # Compute features
    features = _compute_features(times, preds, cycle_time)
    
    # Normalize features
    bl_norm = _normalize(features['bottom_level'])
    td_norm = _normalize(features['total_downstream'])
    ns_norm = _normalize(features['num_succ'])
    cr_norm = _normalize(features['critical_ratio'])
    
    # Base Weights: 
    # We want to prioritize tasks that are critical (high BL) and have high downstream impact (High CR).
    # The Best-Fit logic in _greedy_assign_best_fit handles the packing efficiency,
    # so the priority score can focus more on structural criticality.
    base_weights = [
        (1.0, 0.5, 0.0, 0.5),  # BL, TD, NS, CR
        (1.0, 0.0, 0.0, 1.0),  # BL, CR
        (0.5, 0.5, 0.0, 1.0),  # BL, TD, CR
        (1.0, 0.5, 0.5, 0.5),  # All
        (1.0, 0.0, 0.0, 0.0),  # Pure BL
        (0.0, 0.0, 0.0, 1.0),  # Pure CR
    ]
    
    best_score = np.zeros(n)
    best_stations = float('inf')
    
    # Evaluate base weights
    for w in base_weights:
        if time.time() > deadline:
            break
        score = (w[0] * bl_norm + w[1] * td_norm + w[2] * ns_norm + w[3] * cr_norm)
        st = _greedy_assign_best_fit(times, preds, cycle_time, score)
        if st < best_stations:
            best_stations = st
            best_score = score.copy()
            # If we hit the theoretical lower bound, we can stop early
            total_time = np.sum(times)
            lower_bound = int(np.ceil(total_time / cycle_time))
            if best_stations == lower_bound:
                return best_score

    # Local Search: Perturb the best weights found
    if time.time() < deadline - 5.0:
        # Identify the best base weight vector to start perturbation
        best_w = [1.0, 0.5, 0.0, 0.5]
        for w in base_weights:
            score = (w[0] * bl_norm + w[1] * td_norm + w[2] * ns_norm + w[3] * cr_norm)
            st = _greedy_assign_best_fit(times, preds, cycle_time, score)
            if st == best_stations:
                best_w = list(w)
                break
        
        rng = np.random.default_rng(42)
        n_features = 4
        step_sizes = [0.1, 0.05, 0.01]
        
        for step in step_sizes:
            if time.time() > deadline - 2.0:
                break
            for _ in range(30):
                if time.time() > deadline - 2.0:
                    break
                perturbation = rng.uniform(-step, step, n_features)
                new_w = np.array(best_w) + perturbation
                new_w = np.clip(new_w, 0.0, 2.0)
                
                score = (new_w[0] * bl_norm + new_w[1] * td_norm + new_w[2] * ns_norm + new_w[3] * cr_norm)
                st = _greedy_assign_best_fit(times, preds, cycle_time, score)
                
                if st < best_stations:
                    best_stations = st
                    best_score = score.copy()
                    best_w = new_w.tolist()
                    
                    total_time = np.sum(times)
                    lower_bound = int(np.ceil(total_time / cycle_time))
                    if best_stations == lower_bound:
                        return best_score

    return best_score
# EVOLVE-BLOCK-END
