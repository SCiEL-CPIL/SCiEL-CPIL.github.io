"""Assembly line balancing with min-cut bottleneck priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time as _time

_TIME_BUDGET = 150.0
_DEADLINE_MARGIN = 20.0  # safety margin before 150s kill


def _max_flow_dinic(n_nodes, source, sink, adj, cap):
    """Dinic's max-flow. adj: list of lists of (v, edge_idx_in_adj[u]); cap: list of capacities.
    Returns (flow, min_cut_reachable_set)."""
    flow = 0
    INF = float('inf')
    while True:
        # BFS levels
        level = [-1] * n_nodes
        level[source] = 0
        q = deque([source])
        while q:
            u = q.popleft()
            for v, _ in adj[u]:
                if level[v] < 0 and cap[u][v] > 0:
                    level[v] = level[u] + 1
                    q.append(v)
        if level[sink] < 0:
            break
        # DFS with current arc
        it = [0] * n_nodes
        def dfs(u, f):
            if u == sink:
                return f
            while it[u] < len(adj[u]):
                v, _ = adj[u][it[u]]
                if level[v] == level[u] + 1 and cap[u][v] > 0:
                    ret = dfs(v, min(f, cap[u][v]))
                    if ret > 0:
                        cap[u][v] -= ret
                        cap[v][u] += ret
                        return ret
                it[u] += 1
            return 0
        while True:
            pushed = dfs(source, INF)
            if pushed == 0:
                break
            flow += pushed
    # Reachable set from source in residual
    visited = [False] * n_nodes
    visited[source] = True
    q = deque([source])
    while q:
        u = q.popleft()
        for v, _ in adj[u]:
            if not visited[v] and cap[u][v] > 0:
                visited[v] = True
                q.append(v)
    return flow, visited


def _greedy_balance(times, predecessors, cycle_time, priority_scores):
    """
    Simulate the greedy station filling process.
    Returns number of stations used.
    """
    n = len(times)
    if n == 0:
        return 0
    
    # Build successor list and track remaining work
    successors = [set() for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            if pred not in successors[pred]:
                successors[pred].add(i)
                in_degree[i] += 1
    
    # Topological order for processing
    topo = []
    indeg = list(in_degree)
    q = deque([i for i in range(n) if indeg[i] == 0])
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    
    # Greedy assignment
    assigned = [False] * n
    stations = 0
    station_time = 0.0
    
    # Priority queue: (priority, task_id) - we want highest priority first
    # Use negative for min-heap behavior
    import heapq
    pq = []
    
    # Initially, tasks with no predecessors are eligible
    for i in range(n):
        if in_degree[i] == 0:
            heapq.heappush(pq, (-priority_scores[i], i))
    
    while pq:
        # Get highest priority eligible task
        neg_prio, task = heapq.heappop(pq)
        if assigned[task]:
            continue
        
        # Check if task fits in current station
        if station_time + times[task] > cycle_time:
            # Start new station
            stations += 1
            station_time = 0.0
        
        # Assign task to current station
        assigned[task] = True
        station_time += times[task]
        
        # Update eligible tasks (successors whose predecessors are all assigned)
        for succ in successors[task]:
            # Check if all predecessors of succ are assigned
            all_preds_done = all(assigned[p] for p in predecessors[succ])
            if all_preds_done:
                heapq.heappush(pq, (-priority_scores[succ], succ))
    
    if station_time > 0:
        stations += 1
    
    return stations


def _compute_task_sensitivity(times, predecessors, cycle_time, base_priority):
    """
    For each task, compute how much removing it from the critical path
    affects the overall structure. This is a proxy for bottleneck importance.
    """
    n = len(times)
    if n == 0:
        return np.zeros(n)
    
    successors = [set() for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            if pred not in successors[pred]:
                successors[pred].add(i)
                in_degree[i] += 1
    
    # Topological sort
    topo = []
    indeg = list(in_degree)
    q = deque([i for i in range(n) if indeg[i] == 0])
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    
    # Longest path from any source to each node
    dist_start = [0.0] * n
    for u in topo:
        for v in successors[u]:
            if dist_start[v] < dist_start[u] + times[u]:
                dist_start[v] = dist_start[u] + times[u]
    
    # Longest path from each node to any sink
    dist_end = [0.0] * n
    reverse_topo = list(reversed(topo))
    for u in reverse_topo:
        for pred in predecessors[u]:
            if dist_end[pred] < dist_end[u] + times[u]:
                dist_end[pred] = dist_end[u] + times[u]
    
    # Critical path length through each task
    cp_through = [dist_start[i] + times[i] + dist_end[i] for i in range(n)]
    max_cp = max(cp_through) if cp_through else 1.0
    
    # Sensitivity: how much does this task contribute to the critical path?
    # A task with high cp_through is more important
    # Also consider: number of paths that go through this task
    # For simplicity, use cp_through / max_cp as a normalized importance
    
    sensitivity = np.array(cp_through) / max_cp if max_cp > 0 else np.zeros(n)
    
    return sensitivity


def _build_flow_network(times, predecessors):
    """
    Build a flow network for min-cut analysis.
    Returns (n_nodes, S, T, adj, cap, successors, in_degree)
    """
    n = len(times)
    successors = [set() for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            if pred not in successors[pred]:
                successors[pred].add(i)
                in_degree[i] += 1
    
    S = n
    T = n + 1
    n_nodes = n + 2
    
    adj = [[] for _ in range(n_nodes)]
    cap = [[0.0] * n_nodes for _ in range(n_nodes)]
    
    def add_edge(u, v, c):
        adj[u].append((v, len(adj[u])))
        cap[u][v] += c
        adj[v].append((u, len(adj[v])))
    
    # S -> i with cap = time_i
    for i in range(n):
        add_edge(S, i, float(times[i]))
    
    # i -> j (successor) with cap = INF
    INF_CAP = 1e9
    for i in range(n):
        for j in successors[i]:
            add_edge(i, j, INF_CAP)
    
    # i -> T with cap = time_i
    for i in range(n):
        add_edge(i, T, float(times[i]))
    
    return n_nodes, S, T, adj, cap, successors, in_degree


def _min_cut_priority(times, predecessors, cycle_time):
    """
    Compute priority scores using min-cut bottleneck analysis combined
    with critical path and sensitivity features.
    """
    n = len(times)
    if n == 0:
        return np.array([])
    
    # Build flow network
    n_nodes, S, T, adj, cap, successors, in_degree = _build_flow_network(times, predecessors)
    
    # Compute max flow and min cut
    flow_val, reachable = _max_flow_dinic(n_nodes, S, T, adj, cap)
    
    # Identify bottleneck tasks from min-cut
    bottleneck_score = np.zeros(n)
    for i in range(n):
        if reachable[i] and not reachable[T]:
            bottleneck_score[i] = times[i]
        elif not reachable[i] and reachable[S]:
            bottleneck_score[i] = times[i] * 0.7
    
    # Critical path features
    topo = []
    indeg = list(in_degree)
    q = deque([i for i in range(n) if indeg[i] == 0])
    while q:
        u = q.popleft()
        topo.append(u)
        for v in successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    
    dist_start = [0.0] * n
    for u in topo:
        for v in successors[u]:
            if dist_start[v] < dist_start[u] + times[u]:
                dist_start[v] = dist_start[u] + times[u]
    
    dist_end = [0.0] * n
    reverse_topo = list(reversed(topo))
    for u in reverse_topo:
        for pred in predecessors[u]:
            if dist_end[pred] < dist_end[u] + times[u]:
                dist_end[pred] = dist_end[u] + times[u]
    
    cp = np.array([dist_start[i] + times[i] + dist_end[i] for i in range(n)])
    depth = np.array(dist_start)
    
    # Task sensitivity (importance in critical path)
    sensitivity = _compute_task_sensitivity(times, predecessors, cycle_time, cp)
    
    # Number of successors and predecessors
    num_succ = np.array([len(successors[i]) for i in range(n)])
    num_pred = np.array([in_degree[i] for i in range(n)])
    
    # Normalize all features
    def normalize(arr):
        if arr.max() > 0:
            return arr / arr.max()
        return arr
    
    cp_norm = normalize(cp)
    bn_norm = normalize(bottleneck_score)
    time_norm = normalize(times.astype(float))
    depth_norm = normalize(depth)
    sens_norm = normalize(sensitivity)
    succ_norm = normalize(num_succ.astype(float))
    pred_norm = normalize(num_pred.astype(float))
    
    # Weighted combination with emphasis on structural bottleneck
    priority_scores = (1.2 * cp_norm + 
                       1.0 * bn_norm + 
                       0.8 * sens_norm +
                       0.6 * time_norm +
                       0.4 * succ_norm -
                       0.3 * depth_norm -
                       0.2 * pred_norm)
    
    # Add small tie-breaker
    priority_scores += np.arange(n) * 1e-9
    
    return priority_scores


def _tune_weights_with_greedy(times, predecessors, cycle_time, base_features, base_weights, max_iters=50, time_limit=30.0):
    """
    Use greedy balance simulation to tune weight combination.
    base_features: dict of feature name -> array
    base_weights: dict of feature name -> weight
    Returns best weights found.
    """
    start_time = _time.time()
    n = len(times)
    
    # Current best
    best_weights = dict(base_weights)
    current_priority = np.zeros(n)
    for name, w in base_weights.items():
        current_priority += w * base_features[name]
    best_stations = _greedy_balance(times, predecessors, cycle_time, current_priority)
    
    # Coordinate descent on weights
    feature_names = list(base_weights.keys())
    for iteration in range(max_iters):
        if _time.time() - start_time > time_limit:
            break
        
        improved = False
        for name in feature_names:
            # Try small perturbations of this weight
            current_w = best_weights[name]
            for delta in [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]:
                new_w = max(0.0, current_w + delta)
                if abs(new_w - current_w) < 1e-6:
                    continue
                
                test_weights = dict(best_weights)
                test_weights[name] = new_w
                
                test_priority = np.zeros(n)
                for fname, w in test_weights.items():
                    test_priority += w * base_features[fname]
                
                test_stations = _greedy_balance(times, predecessors, cycle_time, test_priority)
                
                if test_stations < best_stations:
                    best_weights = test_weights
                    best_stations = test_stations
                    improved = True
        
        if not improved:
            break
    
    return best_weights, best_stations


def priority(instance):
    """
    Scores each assembly task using min-cut bottleneck heuristic combined
    with critical path analysis and weight tuning via greedy simulation.
    """
    start_time = _time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])

    # Time guard
    if _time.time() - start_time > _TIME_BUDGET - _DEADLINE_MARGIN:
        return times.astype(float)

    try:
        # Build features
        n_nodes, S, T, adj, cap, successors, in_degree = _build_flow_network(times, predecessors)
        flow_val, reachable = _max_flow_dinic(n_nodes, S, T, adj, cap)
        
        bottleneck_score = np.zeros(n)
        for i in range(n):
            if reachable[i] and not reachable[T]:
                bottleneck_score[i] = times[i]
            elif not reachable[i] and reachable[S]:
                bottleneck_score[i] = times[i] * 0.7
        
        topo = []
        indeg = list(in_degree)
        q = deque([i for i in range(n) if indeg[i] == 0])
        while q:
            u = q.popleft()
            topo.append(u)
            for v in successors[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        
        dist_start = [0.0] * n
        for u in topo:
            for v in successors[u]:
                if dist_start[v] < dist_start[u] + times[u]:
                    dist_start[v] = dist_start[u] + times[u]
        
        dist_end = [0.0] * n
        reverse_topo = list(reversed(topo))
        for u in reverse_topo:
            for pred in predecessors[u]:
                if dist_end[pred] < dist_end[u] + times[u]:
                    dist_end[pred] = dist_end[u] + times[u]
        
        cp = np.array([dist_start[i] + times[i] + dist_end[i] for i in range(n)])
        depth = np.array(dist_start)
        sensitivity = _compute_task_sensitivity(times, predecessors, cycle_time, cp)
        num_succ = np.array([len(successors[i]) for i in range(n)])
        num_pred = np.array([in_degree[i] for i in range(n)])
        
        def normalize(arr):
            if arr.max() > 0:
                return arr / arr.max()
            return arr
        
        features = {
            'cp': normalize(cp),
            'bottleneck': normalize(bottleneck_score),
            'sensitivity': normalize(sensitivity),
            'time': normalize(times.astype(float)),
            'depth': normalize(depth),
            'succ': normalize(num_succ.astype(float)),
            'pred': normalize(num_pred.astype(float)),
        }
        
        # Base weights
        base_weights = {
            'cp': 1.2,
            'bottleneck': 1.0,
            'sensitivity': 0.8,
            'time': 0.6,
            'depth': -0.3,
            'succ': 0.4,
            'pred': -0.2,
        }
        
        # Check time budget for tuning
        elapsed = _time.time() - start_time
        remaining = _TIME_BUDGET - _DEADLINE_MARGIN - elapsed
        
        if remaining > 5.0:
            # Try to tune weights
            tuned_weights, tuned_stations = _tune_weights_with_greedy(
                times, predecessors, cycle_time, features, base_weights,
                max_iters=min(30, int(remaining / 2)),
                time_limit=min(25.0, remaining * 0.8)
            )
            best_weights = tuned_weights
        else:
            best_weights = base_weights
        
        # Compute final priority
        priority_scores = np.zeros(n)
        for name, w in best_weights.items():
            priority_scores += w * features[name]
        
        # Tie-breaker
        priority_scores += np.arange(n) * 1e-9
        
        return priority_scores
        
    except Exception:
        # Fallback to simple time-based priority
        return times.astype(float)
# EVOLVE-BLOCK-END
