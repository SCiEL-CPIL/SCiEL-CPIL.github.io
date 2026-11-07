"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

_START_TIME = time.time()
DEADLINE = _START_TIME + 85.0  # safety margin before 90s kill

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def build_adjacency(edges):
    n = P['n']
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def check_hamiltonian_necessary_conditions(adj, n):
    """Check necessary conditions for Hamiltonian cycle existence."""
    for v in range(n):
        if len(adj[v]) < 2:
            return False
    # Check connectivity
    visited = [False] * n
    visited[0] = True
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                stack.append(v)
    if not all(visited):
        return False
    return True

def find_hamiltonian_cycle_fast(adj, n, start=0, deadline=None, edge_set=None):
    """
    Find a Hamiltonian cycle using Warnsdorff's rule + connectivity pruning.
    edge_set: set of (u,v) keys for O(1) edge lookup if provided.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    if not check_hamiltonian_necessary_conditions(adj, n):
        return None
    
    # Build edge set for fast lookup if not provided
    if edge_set is None:
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                edge_set.add(_edge_key(u, v))
    
    visited = [False] * n
    path = [start]
    visited[start] = True
    sys.setrecursionlimit(10000)
    
    # Precompute adjacency as lists for speed
    adj_list = {v: sorted(adj[v]) for v in range(n)}
    
    def get_unvisited_neighbors(v, current):
        """Get unvisited neighbors of v, prioritized by Warnsdorff's rule."""
        candidates = []
        for nb in adj_list[v]:
            if nb == start and len(path) < n:
                continue  # Can't go back to start until end
            if not visited[nb]:
                # Count unvisited neighbors of nb (Warnsdorff)
                cnt = 0
                for nb2 in adj_list[nb]:
                    if not visited[nb2] and nb2 != start:
                        cnt += 1
                candidates.append((cnt, nb))
        # Sort by fewest unvisited neighbors first (most constrained)
        candidates.sort(key=lambda x: x[0])
        return [nb for _, nb in candidates]
    
    def is_connected_unvisited(unvisited_set, current):
        """Check if unvisited vertices (plus current) form a connected subgraph."""
        if not unvisited_set:
            return True
        # BFS/DFS from current through unvisited vertices
        visited_local = set()
        stack = [current]
        visited_local.add(current)
        while stack:
            u = stack.pop()
            for v in adj_list[u]:
                if v in unvisited_set and v not in visited_local:
                    visited_local.add(v)
                    stack.append(v)
        return visited_local >= (unvisited_set | {current})
    
    def dfs(current, path_len):
        if deadline is not None and time.time() > deadline:
            return None
        if path_len == n:
            if _edge_key(current, start) in edge_set:
                return list(path)
            return None
        
        # Connectivity pruning: check unvisited vertices are connected to current
        unvisited = set()
        for v in range(n):
            if not visited[v] and v != start:
                unvisited.add(v)
        if unvisited:
            if not is_connected_unvisited(unvisited, current):
                return None
        
        # Degree pruning: each unvisited vertex must have >= 2 available neighbors
        for v in unvisited:
            avail = 0
            for nb in adj_list[v]:
                if not visited[nb] or nb == current or nb == start:
                    avail += 1
            if avail < 2:
                return None
        
        # Get next candidates with Warnsdorff ordering
        candidates = get_unvisited_neighbors(current, current)
        for nxt in candidates:
            visited[nxt] = True
            path.append(nxt)
            result = dfs(nxt, path_len + 1)
            if result is not None:
                return result
            path.pop()
            visited[nxt] = False
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
    adj = build_adjacency(edges)
    
    cycles = []
    used_edges = set()
    
    # Theoretical max: floor(len(edges) / n) = floor(70/21) = 3
    max_cycles_to_try = 3
    
    for cycle_idx in range(max_cycles_to_try):
        elapsed = time.time() - _START_TIME
        if elapsed > 82.0:  # Leave 8s safety margin
            break
        
        # Build subgraph of unused edges
        remaining_edges = []
        for e in edges:
            key = _edge_key(e[0], e[1])
            if key not in used_edges:
                remaining_edges.append(e)
        
        # Check if remaining graph can support a Hamiltonian cycle
        rem_adj = build_adjacency(remaining_edges)
        if not check_hamiltonian_necessary_conditions(rem_adj, n):
            break
        
        # Deadline for this cycle search
        deadline = _START_TIME + 82.0
        cycle = None
        
        # Try multiple starting vertices
        for start_v in range(n):
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle_fast(rem_adj, n, start=start_v, deadline=deadline, edge_set=all_edge_keys - used_edges)
            if cycle is not None:
                break
        
        if cycle is None:
            break
        
        # Mark edges as used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            key = _edge_key(u, v)
            used_edges.add(key)
        
        cycles.append(cycle)
    
    # Ensure we have at least 1 cycle
    if not cycles:
        return [list(range(n))]
    
    # Convert to proper format
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
