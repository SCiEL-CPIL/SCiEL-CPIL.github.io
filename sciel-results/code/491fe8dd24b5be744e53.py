"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

_START_TIME = time.time()
GLOBAL_DEADLINE = _START_TIME + 78.0  # Safety margin before 90s kill

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def find_hamiltonian_cycle(adj, n, start, edge_set, deadline, rng=None):
    """
    Optimized backtracking search for a Hamiltonian cycle with strong pruning.
    Uses bitmasks for visited set and precomputed neighbor lists.
    """
    if time.time() > deadline:
        return None
    
    if rng is None:
        rng = random.Random(42)
    
    # Precompute neighbor lists as sorted lists
    adj_list = [sorted(neigh) for neigh in adj]
    
    # Precompute edge lookup as set of frozensets for O(1) lookup
    edge_set_frozenset = set()
    for u, v in edge_set:
        edge_set_frozenset.add(frozenset((u, v)))
    
    def has_edge(u, v):
        return frozenset((u, v)) in edge_set_frozenset
    
    visited = [False] * n
    visited[start] = True
    path = [start]
    
    # Count of unvisited vertices
    unvisited_count = n - 1
    
    # For each vertex, precompute which neighbors are in edge_set
    valid_neighbors = [set() for _ in range(n)]
    for u in range(n):
        for v in adj_list[u]:
            if has_edge(u, v):
                valid_neighbors[u].add(v)
    
    def dfs(current, path_len):
        nonlocal visited, path, unvisited_count
        
        if time.time() > deadline:
            return None
        
        if path_len == n:
            if has_edge(current, start):
                return list(path)
            return None
        
        # Get candidates: unvisited neighbors of current (excluding start unless last)
        candidates = []
        for nb in valid_neighbors[current]:
            if nb == start:
                continue
            if not visited[nb]:
                # Warnsdorff: count unvisited neighbors of nb
                cnt = 0
                for nb2 in valid_neighbors[nb]:
                    if not visited[nb2] and nb2 != start:
                        cnt += 1
                candidates.append((cnt, nb))
        
        if not candidates:
            return None
        
        # Sort by Warnsdorff's rule
        candidates.sort(key=lambda x: x[0])
        
        for _, nxt in candidates:
            if visited[nxt]:
                continue
            
            # Pruning: check if taking nxt isolates any unvisited vertex
            # A vertex is isolated if all its neighbors are visited (and it's not start)
            visited[nxt] = True
            path.append(nxt)
            unvisited_count -= 1
            
            valid = True
            if unvisited_count > 0:
                # Check all unvisited vertices (except start) have at least one unvisited neighbor or neighbor is start
                for uv in range(n):
                    if not visited[uv] and uv != start:
                        deg = 0
                        for nb in valid_neighbors[uv]:
                            if not visited[nb]:
                                deg += 1
                        if deg == 0 and not has_edge(uv, start):
                            valid = False
                            break
            
            if valid:
                result = dfs(nxt, path_len + 1)
                if result is not None:
                    return result
            
            # Backtrack
            visited[nxt] = False
            path.pop()
            unvisited_count += 1
        
        return None
    
    return dfs(start, 1)

def construct():
    n = P['n']
    edges = P['edges']
    
    # Precompute all edge keys
    all_edge_keys = set()
    for u, v in edges:
        all_edge_keys.add(_edge_key(u, v))
    
    # Build initial adjacency
    adj = build_adjacency(edges, n)
    
    cycles = []
    used_edges = set()
    
    max_cycles_to_try = 3
    
    for cycle_idx in range(max_cycles_to_try):
        if time.time() > GLOBAL_DEADLINE:
            break
        
        # Build subgraph of unused edges
        rem_adj = [set() for _ in range(n)]
        for u, v in edges:
            key = _edge_key(u, v)
            if key not in used_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
        
        # Quick check: min degree >= 2
        min_deg = min(len(neigh) for neigh in rem_adj)
        if min_deg < 2:
            break
            
        # Check connectivity of remaining graph
        visited_conn = [False] * n
        stack_conn = [0]
        visited_conn[0] = True
        while stack_conn:
            u = stack_conn.pop()
            for v in rem_adj[u]:
                if not visited_conn[v]:
                    visited_conn[v] = True
                    stack_conn.append(v)
        if not all(visited_conn):
            break
        
        # Deadline for this cycle search
        remaining_time = GLOBAL_DEADLINE - time.time()
        cycles_left = max_cycles_to_try - cycle_idx
        deadline = min(GLOBAL_DEADLINE, time.time() + remaining_time / cycles_left * 0.85)
        
        cycle = None
        
        # Try starting vertices in a strategic order
        # Start with vertex 0, then try others if needed
        start_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        
        for start_v in start_order:
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle(rem_adj, n, start_v, all_edge_keys - used_edges, deadline)
            if cycle is not None:
                break
        
        if cycle is None:
            break
        
        # Mark edges as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            key = _edge_key(u, v)
            if key not in all_edge_keys:
                break
            used_edges.add(key)
        
        cycles.append(cycle)
        
        if len(cycles) == 3:
            break

    # Ensure we have at least 1 cycle
    if not cycles:
        adj_full = build_adjacency(edges, n)
        cycle = find_hamiltonian_cycle(adj_full, n, 0, all_edge_keys, min(GLOBAL_DEADLINE, time.time() + 5.0))
        if cycle:
            return [cycle]
        else:
            # Last resort: find any 3-cycle
            for u, v in edges:
                for w in range(n):
                    if w != u and w != v:
                        if _edge_key(u, v) in all_edge_keys and _edge_key(v, w) in all_edge_keys and _edge_key(w, u) in all_edge_keys:
                            return [[int(u), int(v), int(w)]]
            return [[0, 1, 2]]

    # Convert to proper format
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
