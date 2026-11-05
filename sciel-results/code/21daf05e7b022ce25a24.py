"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _greedy_assign(times, preds, cycle_time, priority_scores):
    """Simulate the greedy station filling with given priority scores."""
    n = len(times)
    if n == 0:
        return 0

    # Precompute successors for each task (which tasks depend on it)
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
        # Fill current station
        while True:
            # Find best eligible task that fits
            best_task = -1
            best_prio = -np.inf
            for i in eligible_list:
                if times[i] <= cycle_time - load + 1e-9:
                    if priority_scores[i] > best_prio:
                        best_prio = priority_scores[i]
                        best_task = i
            if best_task == -1:
                break
            # Assign best_task
            load += times[best_task]
            remaining -= 1
            # Remove from eligible_list
            eligible_list.remove(best_task)
            # Update successors
            for s in succ[best_task]:
                remaining_preds[s] -= 1
                if remaining_preds[s] == 0:
                    eligible_list.append(s)
    
    return stations


def _compute_features(times, preds, cycle_time):
    """Compute various feature vectors for priority computation."""
    n = len(times)
    if n == 0:
        return {}
    
    # Precompute successors
    succ = [[] for _ in range(n)]
    for i in range(n):
        for p in preds[i]:
            succ[p].append(i)
    
    # Topological order
    in_degree = [len(preds[i]) for i in range(n)]
    queue = [i for i in range(n) if in_degree[i] == 0]
    topo_order = []
    temp_in_degree = in_degree[:]
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        for s in succ[node]:
            temp_in_degree[s] -= 1
            if temp_in_degree[s] == 0:
                queue.append(s)
    
    # Bottom-level rank (longest path to end)
    bottom_level = np.zeros(n, dtype=float)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        max_succ = 0.0
        for s in succ[node]:
            if bottom_level[s] > max_succ:
                max_succ = bottom_level[s]
        bottom_level[node] = times[node] + max_succ
    
    # Number of immediate successors
    num_succ = np.array([len(succ[i]) for i in range(n)], dtype=float)
    
    # Number of predecessors
    num_pred = np.array([len(preds[i]) for i in range(n)], dtype=float)
    
    # Total downstream time (sum of times of all successors, not just longest path)
    # Use topological order to compute
    total_downstream = np.zeros(n, dtype=float)
    for i in reversed(range(len(topo_order))):
        node = topo_order[i]
        total_downstream[node] = times[node]
        for s in succ[node]:
            total_downstream[node] += total_downstream[s]
    
    # Criticality: ratio of bottom_level to cycle_time
    criticality = bottom_level / (cycle_time + 1e-9)
    
    features = {
        'bottom_level': bottom_level,
        'times': times,
        'num_succ': num_succ,
        'num_pred': num_pred,
        'total_downstream': total_downstream,
        'criticality': criticality,
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


def priority(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline with 10s safety margin
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    if n == 0:
        return np.array([])
    
    # Compute all features
    features = _compute_features(times, preds, cycle_time)
    succ = features['succ']
    
    # Normalize features
    bl_norm = _normalize(features['bottom_level'])
    t_norm = _normalize(times)
    ns_norm = _normalize(features['num_succ'])
    np_norm = _normalize(features['num_pred'])
    td_norm = _normalize(features['total_downstream'])
    crit_norm = _normalize(features['criticality'])
    
    # Try multiple weight combinations using grid search
    # Weights: (bl, t, ns, np, td, crit)
    base_weights = [
        (1.0, 0.5, 0.5, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (1.0, 0.5, 0.5, 0.0, 0.5, 0.0),
        (1.0, 0.5, 0.5, 0.0, 0.0, 0.5),
        (1.0, 0.5, 0.5, 0.0, 0.5, 0.5),
        (0.5, 1.0, 0.5, 0.0, 0.5, 0.0),
        (1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        (2.0, 0.5, 0.5, 0.0, 0.0, 0.0),
        (1.0, 0.2, 0.2, 0.0, 0.0, 0.0),
    ]
    
    best_score = np.zeros(n)
    best_stations = float('inf')
    
    # Evaluate base weights
    for w in base_weights:
        if time.time() > deadline:
            break
        score = (w[0] * bl_norm + w[1] * t_norm + w[2] * ns_norm + 
                 w[3] * np_norm + w[4] * td_norm + w[5] * crit_norm)
        st = _greedy_assign(times, preds, cycle_time, score)
        if st < best_stations:
            best_stations = st
            best_score = score.copy()
    
    # Local search: perturb the best weights found
    if time.time() < deadline:
        # Extract best weights by finding which base weight gave best_stations
        # Or just perturb the current best_score
        best_w = [1.0, 0.5, 0.5, 0.0, 0.0, 0.0]  # Default
        
        # Try to find which base weight was best
        for w in base_weights:
            score = (w[0] * bl_norm + w[1] * t_norm + w[2] * ns_norm + 
                     w[3] * np_norm + w[4] * td_norm + w[5] * crit_norm)
            st = _greedy_assign(times, preds, cycle_time, score)
            if st == best_stations:
                best_w = list(w)
                break
        
        # Local search with small perturbations
        n_features = 6
        step_sizes = [0.1, 0.05, 0.02]
        
        for step in step_sizes:
            if time.time() > deadline:
                break
            for _ in range(20):  # 20 iterations per step size
                if time.time() > deadline:
                    break
                # Random perturbation
                perturbation = np.random.uniform(-step, step, n_features)
                new_w = np.array(best_w) + perturbation
                new_w = np.clip(new_w, 0.0, 2.0)  # Keep weights in reasonable range
                
                score = (new_w[0] * bl_norm + new_w[1] * t_norm + new_w[2] * ns_norm + 
                         new_w[3] * np_norm + new_w[4] * td_norm + new_w[5] * crit_norm)
                st = _greedy_assign(times, preds, cycle_time, score)
                
                if st < best_stations:
                    best_stations = st
                    best_score = score.copy()
                    best_w = new_w.tolist()
    
    return best_score
# EVOLVE-BLOCK-END
