"""Rewritten program using a structural change: exact longest-path via Held-Karp on 2-core,
with greedy 2-core peeling, DFS backtracking with pruning, and local 2-opt refinement."""
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _two_core(adj):
    """Return the 2-core of the graph as a set of vertices."""
    n = P['n']
    deg = [len(adj[i]) for i in range(n)]
    in_core = [True] * n
    q = deque([i for i in range(n) if deg[i] < 2])
    while q:
        u = q.popleft()
        if not in_core[u]:
            continue
        in_core[u] = False
        for v in adj[u]:
            if in_core[v]:
                deg[v] -= 1
                if deg[v] < 2:
                    q.append(v)
    return set(i for i in range(n) if in_core[i])

def _greedy_path(adj, start=None):
    """Greedy path construction with a simple heuristic."""
    n = P['n']
    if start is None:
        # Start from a vertex with high degree in the 2-core
        core = _two_core(adj)
        if core:
            start = max(core, key=lambda x: len(adj[x] & core))
        else:
            start = max(range(n), key=lambda x: len(adj[x]))
    
    path = [start]
    seen = {start}
    core = _two_core(adj)
    while True:
        curr = path[-1]
        candidates = [v for v in adj[curr] if v not in seen]
        if not candidates:
            break
        # Prefer vertices in core with higher degree
        def score(v):
            in_core = 1 if v in core else 0
            return (in_core, len(adj[v] & core), -len(adj[v]))
        v = max(candidates, key=score)
        path.append(v)
        seen.add(v)
    return path

def _dfs_longest(adj, time_limit, deadline):
    """DFS with backtracking to find longest path, with pruning."""
    n = P['n']
    core = _two_core(adj)
    # Work on the induced subgraph of the 2-core if it's small enough
    if len(core) <= 30:
        vertices = sorted(core)
        idx_map = {v: i for i, v in enumerate(vertices)}
        m = len(vertices)
        # Build adjacency in compressed indices
        adj_comp = [set() for _ in range(m)]
        for i, u in enumerate(vertices):
            for v in adj[u]:
                if v in idx_map:
                    adj_comp[i].add(idx_map[v])
        # Held-Karp style DP for longest path
        # dp[mask][last] = length of path using exactly vertices in mask, ending at last
        # This is O(2^m * m^2) which is feasible for m <= 20-25
        if m <= 22:
            return _held_karp(adj_comp, vertices, deadline)
    
    # Fall back to DFS with pruning
    best_path = []
    best_len = 0
    
    # Try starting from each vertex in core (or all if core is empty)
    start_candidates = sorted(core) if core else list(range(n))
    # Limit starts to keep within time
    max_starts = min(len(start_candidates), 100)
    
    for start in start_candidates[:max_starts]:
        if time.time() > deadline:
            break
        path = [start]
        seen = {start}
        
        def dfs():
            nonlocal best_path, best_len
            if time.time() > deadline:
                return
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            curr = path[-1]
            # Pruning: if remaining possible vertices + current len <= best_len, stop
            # Simple bound: number of unvisited neighbors reachable
            candidates = [v for v in adj[curr] if v not in seen]
            if not candidates:
                return
            
            # Sort by degree to explore promising branches first
            candidates.sort(key=lambda v: -len(adj[v]))
            
            for v in candidates:
                path.append(v)
                seen.add(v)
                dfs()
                path.pop()
                seen.discard(v)
                if time.time() > deadline:
                    return
        
        dfs()
    
    return best_path

def _held_karp(adj_comp, vertices, deadline):
    """Exact longest path via DP on the 2-core if small enough."""
    m = len(vertices)
    if m == 0:
        return []
    
    # dp[mask][last] = max length of path using vertices in mask, ending at last
    # mask is a bitmask of m bits
    full_mask = (1 << m) - 1
    dp = [[-1] * m for _ in range(1 << m)]
    
    # Initialize: paths of length 1
    for i in range(m):
        dp[1 << i][i] = 1
    
    # Build lookup for quick access
    from functools import lru_cache
    
    # Iterate over masks
    for mask in range(1, 1 << m):
        if time.time() > deadline:
            break
        for last in range(m):
            if dp[mask][last] < 0:
                continue
            # Try extending to next vertex
            for nxt in adj_comp[last]:
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                if dp[new_mask][nxt] < dp[mask][last] + 1:
                    dp[new_mask][nxt] = dp[mask][last] + 1
    
    # Find best
    best_len = 0
    best_mask = 0
    best_last = 0
    for mask in range(1, 1 << m):
        for last in range(m):
            if dp[mask][last] > best_len:
                best_len = dp[mask][last]
                best_mask = mask
                best_last = last
    
    if best_len == 0:
        return [vertices[0]]
    
    # Reconstruct path
    path = [best_last]
    mask = best_mask
    last = best_last
    while mask != (1 << last):
        # Find previous vertex
        for prev in adj_comp[last]:
            prev_mask = mask ^ (1 << last)
            if prev_mask & (1 << prev) and dp[prev_mask][prev] == dp[mask][last] - 1:
                path.append(prev)
                mask = prev_mask
                last = prev
                break
        else:
            break
    
    path.reverse()
    return [vertices[i] for i in path]

def _two_opt_refine(path, adj):
    """Try 2-opt improvements on the path."""
    if len(path) < 3:
        return path
    
    n = P['n']
    adj_set = [set(adj[i]) for i in range(n)]
    improved = True
    best_path = path[:]
    
    while improved:
        improved = False
        for i in range(len(best_path) - 1):
            if i + 2 >= len(best_path):
                break
            a, b = best_path[i], best_path[i + 1]
            # Try reversing segment from i+1 to j
            for j in range(i + 2, len(best_path)):
                c, d = best_path[j], best_path[j + 1] if j + 1 < len(best_path) else None
                # Check if removing edge (a,b) and (c,d) and adding (a,c) and (b,d) works
                if d is None:
                    # Reverse from i+1 to end
                    if a in adj_set[c]:
                        new_path = best_path[:i+1] + best_path[i+1:j+1][::-1]
                        if len(set(new_path)) == len(new_path):
                            valid = True
                            for k in range(len(new_path) - 1):
                                if new_path[k+1] not in adj_set[new_path[k]]:
                                    valid = False
                                    break
                            if valid:
                                best_path = new_path
                                improved = True
                                break
                else:
                    # Check edges
                    if a in adj_set[c] and b in adj_set[d]:
                        new_path = best_path[:i+1] + best_path[i+1:j+1][::-1] + best_path[j+1:]
                        if len(set(new_path)) == len(new_path):
                            valid = True
                            for k in range(len(new_path) - 1):
                                if new_path[k+1] not in adj_set[new_path[k]]:
                                    valid = False
                                    break
                            if valid:
                                best_path = new_path
                                improved = True
                                break
            if improved:
                break
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    adj = _build_adj()
    
    # Step 1: Greedy initial path
    best_path = _greedy_path(adj)
    
    # Step 2: Try DFS-based longest path search with time limit
    if time.time() < deadline - 10.0:
        dfs_path = _dfs_longest(adj, time.time(), deadline - 5.0)
        if len(dfs_path) > len(best_path):
            best_path = dfs_path
    
    # Step 3: 2-opt refinement
    if time.time() < deadline - 2.0:
        refined = _two_opt_refine(best_path, adj)
        if len(refined) > len(best_path):
            best_path = refined
    
    # Ensure validity
    if not best_path:
        best_path = [0]
    
    # Final validation
    adj_set = [set(adj[i]) for i in range(P['n'])]
    valid = True
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in adj_set[best_path[i]]:
            valid = False
            break
    if len(set(best_path)) != len(best_path):
        valid = False
    
    if not valid:
        # Fallback to a simple valid path
        best_path = _greedy_path(adj)
    
    return best_path
# EVOLVE-BLOCK-END
