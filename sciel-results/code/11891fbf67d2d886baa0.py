import numpy as np
from collections import defaultdict, deque
import time

def priority(instance):
    start_time = time.time()
    deadline = start_time + 10.0  # 10 second deadline, well within 120s
    
    times = np.asarray(instance["times"], dtype=float)
    preds = instance["predecessors"]
    cycle_time = instance["cycle_time"]
    n = len(times)
    
    if n == 0:
        return np.array([])
    
    # Build successors graph
    succs = defaultdict(list)
    for i in range(n):
        for p in preds[i]:
            succs[p].append(i)
    
    # Compute in-degree for topological sort
    in_degree = np.zeros(n, dtype=int)
    for i in range(n):
        in_degree[i] = len(preds[i])
    
    # Topological sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    topo_order = []
    while queue:
        if time.time() > deadline:
            break
        u = queue.popleft()
        topo_order.append(u)
        for v in succs[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    # If topo_order doesn't have all nodes, there's a cycle. Handle by adding remaining nodes.
    if len(topo_order) < n:
        remaining = set(range(n)) - set(topo_order)
        topo_order.extend(sorted(remaining))
    
    # Compute critical path (forward) in reverse topological order
    cp = np.zeros(n)
    for u in reversed(topo_order):
        if time.time() > deadline:
            break
        max_succ_cp = 0.0
        for v in succs[u]:
            if cp[v] > max_succ_cp:
                max_succ_cp = cp[v]
        cp[u] = times[u] + max_succ_cp
    
    # Compute number of transitive successors and sum of their times for each node
    # Using reverse topological order allows dynamic programming for sums, but BFS is safer for correctness if graph is not a DAG (though it is).
    # However, for 30 nodes, BFS is fine. Let's optimize by computing in reverse topo order if possible.
    # Actually, since we have `topo_order`, we can compute `downstream_time` and `trans_succ_count` more efficiently using DP in reverse topological order.
    
    downstream_time = np.zeros(n)
    trans_succ_count = np.zeros(n, dtype=int)
    
    for u in reversed(topo_order):
        if time.time() > deadline:
            break
        max_cp_succ = 0.0
        sum_time_succ = 0.0
        count_succ = 0
        
        # We need to be careful: `cp[u]` was computed as `times[u] + max(cp[v])`.
        # `downstream_time[u]` should be sum of times of all transitive successors.
        # This is NOT simply sum of `downstream_time[v]` because of overlaps in DAGs.
        # So we must use BFS or bitsets for exact count/sum of unique successors.
        # Given n=30, we can use bitmasks!
        
        # Let's use bitmasks for exact transitive closure properties.
        # This is much faster and allows exact counting and summing if we precompute.
        
        # Re-initializing for bitmask approach as it's cleaner for n<=30 (fits in int64)
        pass 

    # Let's stick to the BFS approach but maybe tweak the priority formula.
    # The current formula `cp * (1 + trans_succ_count)` might over-prioritize wide nodes.
    # Let's try `cp + trans_succ_count * times[u]`? No.
    # Let's try `cp + downstream_time` (sum of times of successors).
    
    # Let's re-compute downstream_time using BFS as well, or just modify the existing loop.
    
    # To minimize changes and risk, let's just adjust the priority formula using the existing `trans_succ_count` and `cp`.
    # Maybe `cp + trans_succ_count`?
    # Or `cp * (1 + trans_succ_count * 0.1)`?
    
    # Let's try a different heuristic: Rank Position Value (RPV) is sum of times of node and all successors.
    # RPV[u] = times[u] + sum(times[v] for v in transitive_successors(u))
    # This is a very strong heuristic for ALB.
    
    # Let's compute RPV.
    rpv = np.zeros(n)
    for u in range(n):
        if time.time() > deadline:
            break
        visited = set()
        dq = deque(succs[u])
        total_time = 0.0
        while dq:
            v = dq.popleft()
            if v not in visited:
                visited.add(v)
                total_time += times[v]
                for w in succs[v]:
                    if w not in visited:
                        dq.append(w)
        rpv[u] = times[u] + total_time
        
    # Priority: RPV is often superior to CP alone.
    # Let's combine RPV and CP.
    # priority = rpv + cp?
    # Or just rpv?
    
    # Let's use RPV as the primary score.
    priority_scores = rpv
    
    return priority_scores
