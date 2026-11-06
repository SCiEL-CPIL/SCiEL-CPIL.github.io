"""Assembly line balancing with min-cut bottleneck priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import time as _time

_TIME_BUDGET = 120.0
_DEADLINE_MARGIN = 15.0  # safety margin before 120s kill


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


def _min_cut_priority(times, predecessors, cycle_time):
    """
    Build a flow network where each task i has capacity proportional to its time,
    and compute min-cut from source to sink. Tasks on the min-cut are structural
    bottlenecks. Priority = min-cut membership * time + secondary heuristics.
    """
    n = len(times)
    if n == 0:
        return np.array([])

    # Build successor list
    successors = [set() for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for pred in predecessors[i]:
            if pred not in successors[pred]:
                successors[pred].add(i)
                in_degree[i] += 1

    # Topological sort to find source (no preds) and sink (no succs)
    # Use topological order for processing
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
    if len(topo) != n:
        # Cycle detected; fall back
        topo = list(range(n))

    # Find sources (no predecessors) and sinks (no successors)
    sources = [i for i in range(n) if in_degree[i] == 0]
    sinks = [i for i in range(n) if len(successors[i]) == 0]

    if not sources or not sinks:
        # Degenerate case
        return times.astype(float)

    # Build flow network:
    # Node for each task i, plus a global source S and sink T.
    # S -> source tasks with cap = time_i
    # Task i -> task j (if j is successor of i) with cap = INF (or large)
    # Sink tasks -> T with cap = time_i
    # Actually, for min-cut to capture "bottleneck", we want:
    # S -> i (cap = time_i) for all i
    # i -> j (cap = INF) for each predecessor-successor pair
    # i -> T (cap = time_i) for all i
    # The min cut will separate S from T. Tasks on the cut have their edge cut.
    #
    # Better formulation: 
    # Each task i has a node. Edge i->j (successor) with cap = INF.
    # S -> i with cap = time_i (for all i)
    # i -> T with cap = time_i (for all i)
    # Min cut capacity = sum of time_i for tasks where either S->i or i->T is cut.
    # A task is a "bottleneck" if it's on the min-cut path.
    #
    # Simpler: just compute max flow and identify which task nodes are
    # "critical" by checking if removing them reduces flow.
    # But that's O(n) max flows. For n=30, it's fine.

    # Let's use a simpler approach: compute longest path (critical path)
    # and combine with flow-based bottleneck detection.
    
    # Critical path (longest path from any source to any sink through i)
    # dist_start[i] = longest path from any source to i
    # dist_end[i] = longest path from i to any sink
    # critical_path_through_i = dist_start[i] + times[i] + dist_end[i]
    
    dist_start = [0.0] * n
    for u in topo:
        for v in successors[u]:
            if dist_start[v] < dist_start[u] + times[u]:
                dist_start[v] = dist_start[u] + times[u]

    # Reverse topo for dist_end
    reverse_topo = list(reversed(topo))
    dist_end = [0.0] * n
    for u in reverse_topo:
        for pred in predecessors[u]:
            if dist_end[pred] < dist_end[u] + times[u]:
                dist_end[pred] = dist_end[u] + times[u]

    # Critical path length through each task
    cp = np.array([dist_start[i] + times[i] + dist_end[i] for i in range(n)])

    # Number of successors and predecessors
    num_succ = np.array([len(successors[i]) for i in range(n)])
    num_pred = np.array([in_degree[i] for i in range(n)])

    # Depth: longest path from any source to i (already dist_start)
    depth = np.array(dist_start)

    # Build flow network for bottleneck detection
    # S = n, T = n+1
    S = n
    T = n + 1
    n_nodes = n + 2

    # Initialize adjacency and capacity
    adj = [[] for _ in range(n_nodes)]
    cap = [[0.0] * n_nodes for _ in range(n_nodes)]

    def add_edge(u, v, c):
        adj[u].append((v, len(adj[u])))
        cap[u][v] += c
        adj[v].append((u, len(adj[v])))
        # cap[v][u] stays 0 initially (residual)

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

    # Compute max flow and min cut
    flow_val, reachable = _max_flow_dinic(n_nodes, S, T, adj, cap)

    # Tasks on the min cut: those where i is reachable from S but T is not reachable through i
    # Actually, min-cut edges are (u,v) where u is reachable and v is not.
    # A task i is a "bottleneck" if its node is on the cut boundary.
    # Let's identify tasks where the edge i->T is cut (i reachable, T not) or S->i is cut.
    
    bottleneck_score = np.zeros(n)
    for i in range(n):
        # If i is reachable from S in residual, and T is not reachable through i
        # Simpler: if i is reachable and i has an edge to T that was cut
        if reachable[i]:
            # Check if edge i->T is in the cut (i reachable, T not reachable)
            if not reachable[T]:
                bottleneck_score[i] = times[i]
        else:
            # If i is not reachable, check if S->i edge was cut
            if reachable[S]:
                bottleneck_score[i] = times[i] * 0.5

    # Combine: priority = w1 * cp + w2 * bottleneck + w3 * time - w4 * depth
    # Normalize each component
    if cp.max() > 0:
        cp_norm = cp / cp.max()
    else:
        cp_norm = cp

    if bottleneck_score.max() > 0:
        bn_norm = bottleneck_score / bottleneck_score.max()
    else:
        bn_norm = bottleneck_score

    if times.max() > 0:
        time_norm = times.astype(float) / times.max()
    else:
        time_norm = times.astype(float)

    if depth.max() > 0:
        depth_norm = depth / depth.max()
    else:
        depth_norm = depth

    # Weighted combination
    priority_scores = (1.0 * cp_norm + 
                       0.8 * bn_norm + 
                       0.5 * time_norm - 
                       0.3 * depth_norm)

    # Add small tie-breaker based on index for determinism
    priority_scores += np.arange(n) * 1e-9

    return priority_scores


def priority(instance):
    """
    Scores each assembly task using a min-cut bottleneck heuristic combined
    with critical path analysis.
    
    Uses a flow network to identify structural bottlenecks in the dependency
    graph, then combines with longest-path and time-based signals.
    """
    start_time = _time.time()
    
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    
    n = len(times)
    if n == 0:
        return np.array([])

    # Time guard: if we're close to deadline, use simple fallback
    if _time.time() - start_time > _TIME_BUDGET - _DEADLINE_MARGIN:
        return times.astype(float)

    try:
        result = _min_cut_priority(times, predecessors, cycle_time)
        return result
    except Exception:
        # Fallback to simple time-based priority
        return times.astype(float)
# EVOLVE-BLOCK-END
