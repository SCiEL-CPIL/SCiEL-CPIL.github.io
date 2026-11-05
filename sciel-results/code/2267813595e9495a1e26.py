"""Rewritten program for Resource-Constrained Project Scheduling."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque
import time

_START_TIME = None
_TIME_LIMIT = 145.0  # Safety margin before 150s wall-clock kill

def priority(instance):
    """
    Score each activity for the serial schedule-generation scheme.
    Higher score = higher priority (started earlier).
    
    Heuristic components:
    1. Bottom Level (BL): Longest path from node to end.
    2. Total Successors (TS): Number of activities that depend on this one (transitively).
    3. Resource Pressure (RP): Sum of (demand / capacity).
    4. Duration: Longer jobs get slight priority.
    
    Score = w1*BL + w2*TS + w3*RP + w4*Dur
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    durations = np.asarray(instance["durations"], dtype=float)
    n = len(durations)
    
    if n == 0:
        return np.array([])

    demands = instance.get("demands", None)
    capacities = instance.get("capacities", None)
    preds = instance.get("predecessors", [None] * n)

    # 1. Compute Graph Structures
    pred_lists = []
    for p in preds:
        if p is None:
            pred_lists.append([])
        elif isinstance(p, (list, tuple, np.ndarray)):
            pred_lists.append([x for x in p if x is not None])
        else:
            pred_lists.append([p])
            
    successors = [[] for _ in range(n)]
    in_degree = [0] * n
    for i in range(n):
        for p in pred_lists[i]:
            if 0 <= p < n:
                successors[p].append(i)
                in_degree[i] += 1
            
    # Topological Sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in successors[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # If cycle exists, fallback
    if len(top_order) < n:
        top_order = list(range(n))

    # 2. Compute Bottom Level (Longest path to end)
    bottom_level = np.zeros(n)
    for u in reversed(top_order):
        for v in successors[u]:
            candidate = bottom_level[v] + durations[v]
            if candidate > bottom_level[u]:
                bottom_level[u] = candidate
    
    # 3. Compute Total Successors (Transitive)
    # For N=30, O(N^2) or O(N*(N+E)) is very fast.
    # We can use a DP approach if we just want count of reachable nodes.
    # Since it's a DAG, we can compute this in reverse topological order if we use bitsets or just simple sets.
    # Given N is small, simple BFS/DFS from each node is acceptable and less error-prone.
    # However, let's try a DP with sets for efficiency if N was larger, but for 30, simple recursion with memoization or iterative is fine.
    # Let's use a simple iterative approach: 
    # succ_count[i] = size of set of all nodes reachable from i.
    # We can compute this in reverse topological order:
    # Reachable(i) = Union( {j} for j in succ(i) ) U Union( Reachable(j) for j in succ(i) )
    # This is expensive with sets if not careful. 
    # Alternative: Just count immediate successors? No, transitive is better.
    # Let's just do a BFS from each node. 30 nodes * 30 edges max = 900 ops. Trivial.
    
    total_successors = np.zeros(n, dtype=int)
    for i in range(n):
        # BFS from i
        visited = set()
        q = deque([i])
        while q:
            u = q.popleft()
            for v in successors[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        total_successors[i] = len(visited)

    # 4. Compute Resource Metrics
    if demands is None or capacities is None:
        # Fallback: Just BL
        bl_max = np.max(bottom_level)
        if bl_max > 0:
            return bottom_level / bl_max
        return bottom_level

    # Process demands into matrix
    if isinstance(demands, list) and len(demands) > 0:
        if isinstance(demands[0], dict):
            n_resources = len(capacities)
            demand_matrix = np.zeros((n, n_resources), dtype=float)
            for i, d in enumerate(demands):
                for r, val in d.items():
                    if r < n_resources:
                        demand_matrix[i, r] = val
        elif isinstance(demands[0], (list, tuple, np.ndarray)):
            demand_matrix = np.asarray(demands, dtype=float)
            if demand_matrix.shape[0] != n:
                try:
                    demand_matrix = demand_matrix.reshape(n, -1)
                except:
                    demand_matrix = np.zeros((n, 1))
        else:
            demand_matrix = np.zeros((n, 1), dtype=float)
    else:
        demand_matrix = np.zeros((n, 1), dtype=float)
        
    capacities_arr = np.asarray(capacities, dtype=float)
    if len(capacities_arr) == 0:
        capacities_arr = np.array([1.0])
        
    # Ensure dimensions match
    if demand_matrix.shape[1] != len(capacities_arr):
        new_dim = min(demand_matrix.shape[1], len(capacities_arr))
        demand_matrix = demand_matrix[:, :new_dim]
        capacities_arr = capacities_arr[:new_dim]
        
    safe_caps = np.where(capacities_arr > 0, capacities_arr, 1.0)
    
    # Resource Pressure: Sum of ratios
    resource_pressure = np.sum(demand_matrix / safe_caps, axis=1)

    # 5. Normalize and Combine
    bl_max = np.max(bottom_level) if np.max(bottom_level) > 0 else 1.0
    ts_max = np.max(total_successors) if np.max(total_successors) > 0 else 1.0
    rp_max = np.max(resource_pressure) if np.max(resource_pressure) > 0 else 1.0
    dur_max = np.max(durations) if np.max(durations) > 0 else 1.0

    bl_norm = bottom_level / bl_max
    ts_norm = total_successors.astype(float) / ts_max
    rp_norm = resource_pressure / rp_max
    dur_norm = durations / dur_max

    # Weights
    # BL is the primary driver
    # TS helps unblock the graph
    # RP handles resource constraints
    # Dur is a minor tie-breaker
    w_bl = 0.65
    w_ts = 0.20
    w_rp = 0.10
    w_dur = 0.05

    scores = w_bl * bl_norm + w_ts * ts_norm + w_rp * rp_norm + w_dur * dur_norm

    return scores
# EVOLVE-BLOCK-END
