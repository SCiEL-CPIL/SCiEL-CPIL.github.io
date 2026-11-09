"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from collections import deque

def _simulate_stations(times, predecessors, successors, cycle_time, priority_scores, n):
    """Simulate greedy station assignment and return number of stations used."""
    assigned = [False] * n
    remaining_preds = [len(predecessors[i]) for i in range(n)]
    
    station_count = 0
    remaining = n
    
    while remaining > 0:
        station_count += 1
        remaining_time = cycle_time
        # Find eligible tasks (all predecessors assigned)
        eligible = []
        for i in range(n):
            if not assigned[i] and remaining_preds[i] == 0:
                eligible.append(i)
        
        if not eligible:
            break
        
        # Sort by priority (descending) and assign greedily
        eligible.sort(key=lambda i: priority_scores[i], reverse=True)
        
        for i in eligible:
            if not assigned[i] and times[i] <= remaining_time:
                assigned[i] = True
                remaining_time -= times[i]
                remaining -= 1
                # Update successor counts
                for v in successors[i]:
                    remaining_preds[v] -= 1
    
    return station_count

def _compute_features(times, predecessors, successors, n):
    """Compute various features for priority scoring."""
    # Topological order (Kahn's algorithm)
    in_degree = [len(predecessors[i]) for i in range(n)]
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo = []
    indeg_copy = in_degree[:]
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg_copy[v] -= 1
            if indeg_copy[v] == 0:
                queue.append(v)
    
    if len(topo) != n:
        return None
    
    # Longest downstream path (rank)
    rank = np.zeros(n, dtype=float)
    for u in reversed(topo):
        max_succ = 0.0
        for v in successors[u]:
            if rank[v] > max_succ:
                max_succ = rank[v]
        rank[u] = times[u] + max_succ
    
    # Longest path to start
    rank_start = np.zeros(n, dtype=float)
    for u in topo:
        for v in successors[u]:
            val = rank_start[u] + times[u]
            if val > rank_start[v]:
                rank_start[v] = val
    
    # Number of successors (out-degree)
    out_degree = np.array([len(successors[i]) for i in range(n)], dtype=float)
    
    # Number of predecessors (in-degree)
    in_deg_arr = np.array(in_degree, dtype=float)
    
    # Topological position
    topo_pos = np.zeros(n, dtype=float)
    for idx, u in enumerate(topo):
        topo_pos[u] = idx / max(n - 1, 1)
    
    # Successor count weighted by time (total downstream work)
    downstream_work = np.zeros(n, dtype=float)
    for u in reversed(topo):
        total = 0.0
        for v in successors[u]:
            total += times[v] + downstream_work[v]
        downstream_work[u] = total
    
    # Immediate successor count
    imm_succ_count = out_degree
    
    # Number of tasks that can follow (transitive successors)
    trans_succ_count = np.zeros(n, dtype=int)
    for u in reversed(topo):
        count = 0
        for v in successors[u]:
            count += 1 + trans_succ_count[v]
        trans_succ_count[u] = count
    
    return {
        'rank': rank,
        'rank_start': rank_start,
        'out_degree': out_degree,
        'in_degree': in_deg_arr,
        'topo_pos': topo_pos,
        'downstream_work': downstream_work,
        'trans_succ_count': trans_succ_count.astype(float),
        'time': times.copy(),
    }

def _soft_rank(scores, temperature=1.0):
    """Compute a soft ranking that accounts for relative ordering."""
    # Use a smooth approximation: for each task, compute how many others have higher priority
    # This creates a smoother signal than hard ranking
    n = len(scores)
    soft_ranks = np.zeros(n, dtype=float)
    for i in range(n):
        # Count tasks with significantly higher priority
        diffs = scores - scores[i]
        # Soft count: sum of sigmoid(diffs / temperature)
        soft_ranks[i] = np.sum(1.0 / (1.0 + np.exp(-diffs / temperature)))
    return soft_ranks

def priority(instance):
    start_time = _time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    times = np.asarray(instance["times"], dtype=float)
    n = len(times)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    # Build successor list
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        preds = predecessors[i]
        for p in preds:
            successors[p].append(i)
            in_degree[i] += 1
    
    # Compute features
    features = _compute_features(times, predecessors, successors, n)
    if features is None:
        return times.copy()
    
    # Extract features
    rank = features['rank']
    rank_start = features['rank_start']
    out_degree = features['out_degree']
    in_deg_arr = features['in_degree']
    topo_pos = features['topo_pos']
    downstream_work = features['downstream_work']
    trans_succ = features['trans_succ_count']
    time_arr = features['time']
    
    # Normalize all features
    def normalize(arr):
        max_val = np.max(arr) if len(arr) > 0 and np.max(arr) > 0 else 1.0
        min_val = np.min(arr) if len(arr) > 0 else 0.0
        range_val = max_val - min_val
        if range_val < 1e-9:
            return np.zeros_like(arr)
        return (arr - min_val) / range_val
    
    norm_rank = normalize(rank)
    norm_rank_start = normalize(rank_start)
    norm_outdeg = normalize(out_degree)
    norm_indeg = normalize(in_deg_arr)
    norm_topo_pos = normalize(topo_pos)
    norm_downstream_work = normalize(downstream_work)
    norm_trans_succ = normalize(trans_succ)
    norm_time = normalize(time_arr)
    
    # Baseline priority: weighted combination of key features
    # Emphasize rank (longest path) and downstream work
    base_priority = (0.4 * norm_rank + 
                     0.25 * norm_downstream_work + 
                     0.15 * norm_outdeg + 
                     0.1 * norm_rank_start + 
                     0.1 * norm_time)
    
    # Deadline check before local search
    if _time.time() - start_time > DEADLINE - 20.0:
        return base_priority
    
    best_priority = base_priority.copy()
    best_stations = _simulate_stations(times, predecessors, successors, cycle_time, best_priority, n)
    
    # Try a broader set of weight combinations
    # Weights: (rank, downstream_work, outdeg, rank_start, time, trans_succ, indeg, topo_pos)
    weight_options = [
        (0.4, 0.25, 0.15, 0.1, 0.1, 0.0, 0.0, 0.0),
        (0.5, 0.2, 0.15, 0.1, 0.05, 0.0, 0.0, 0.0),
        (0.3, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0),
        (0.45, 0.2, 0.15, 0.1, 0.1, 0.0, 0.0, 0.0),
        (0.35, 0.25, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0),
        (0.5, 0.25, 0.1, 0.1, 0.05, 0.0, 0.0, 0.0),
        (0.4, 0.3, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0),
        (0.3, 0.25, 0.2, 0.1, 0.1, 0.0, 0.05, 0.0),
        (0.4, 0.2, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0),
        (0.35, 0.25, 0.15, 0.1, 0.1, 0.05, 0.0, 0.0),
        (0.45, 0.2, 0.15, 0.1, 0.1, 0.0, 0.0, 0.0),
        (0.5, 0.15, 0.15, 0.1, 0.1, 0.0, 0.0, 0.0),
        (0.4, 0.2, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0),
        (0.3, 0.3, 0.15, 0.1, 0.1, 0.05, 0.0, 0.0),
        (0.35, 0.2, 0.2, 0.1, 0.1, 0.05, 0.0, 0.0),
        (0.4, 0.25, 0.1, 0.1, 0.1, 0.05, 0.0, 0.0),
        (0.5, 0.2, 0.1, 0.1, 0.1, 0.05, 0.0, 0.0),
        (0.45, 0.2, 0.15, 0.1, 0.1, 0.0, 0.0, 0.0),
        (0.3, 0.25, 0.25, 0.1, 0.1, 0.0, 0.0, 0.0),
        (0.35, 0.25, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0),
    ]
    
    for w in weight_options:
        if _time.time() - start_time > DEADLINE - 10.0:
            break
        trial_priority = (w[0] * norm_rank + w[1] * norm_downstream_work + 
                         w[2] * norm_outdeg + w[3] * norm_rank_start + 
                         w[4] * norm_time + w[5] * norm_trans_succ + 
                         w[6] * norm_indeg + w[7] * norm_topo_pos)
        stations = _simulate_stations(times, predecessors, successors, cycle_time, trial_priority, n)
        if stations < best_stations:
            best_stations = stations
            best_priority = trial_priority.copy()
    
    # Try adding bonuses for different task characteristics
    if _time.time() - start_time > DEADLINE - 8.0:
        return best_priority
    
    bonus_configs = [
        ('time', [0.1, 0.2, 0.3, 0.15, 0.25]),
        ('rank', [0.1, 0.2, 0.15, 0.25]),
        ('downstream_work', [0.1, 0.2, 0.15, 0.25]),
        ('outdeg', [0.1, 0.2, 0.15, 0.25]),
    ]
    
    for feat_name, bonus_values in bonus_configs:
        if _time.time() - start_time > DEADLINE - 5.0:
            break
        feat_map = {
            'time': norm_time,
            'rank': norm_rank,
            'downstream_work': norm_downstream_work,
            'outdeg': norm_outdeg,
        }
        feat = feat_map[feat_name]
        for bonus in bonus_values:
            if _time.time() - start_time > DEADLINE - 3.0:
                break
            trial_priority = best_priority + bonus * feat
            stations = _simulate_stations(times, predecessors, successors, cycle_time, trial_priority, n)
            if stations < best_stations:
                best_stations = stations
                best_priority = trial_priority.copy()
    
    # Try soft ranking to smooth out the priority signal
    if _time.time() - start_time > DEADLINE - 4.0:
        return best_priority
    
    for temp in [0.5, 1.0, 2.0, 0.3, 0.7]:
        if _time.time() - start_time > DEADLINE - 2.0:
            break
        soft_r = _soft_rank(best_priority, temperature=temp)
        soft_r_norm = normalize(soft_r)
        trial_priority = 0.5 * best_priority + 0.5 * soft_r_norm
        stations = _simulate_stations(times, predecessors, successors, cycle_time, trial_priority, n)
        if stations < best_stations:
            best_stations = stations
            best_priority = trial_priority.copy()
    
    # Final polish: small perturbations
    if _time.time() - start_time > DEADLINE - 1.5:
        return best_priority
    
    for delta in [-0.05, 0.05, -0.1, 0.1, -0.03, 0.03]:
        if _time.time() - start_time > DEADLINE - 0.5:
            break
        trial_priority = best_priority * (1.0 + delta * norm_time)
        stations = _simulate_stations(times, predecessors, successors, cycle_time, trial_priority, n)
        if stations < best_stations:
            best_stations = stations
            best_priority = trial_priority.copy()
    
    return best_priority
# EVOLVE-BLOCK-END
