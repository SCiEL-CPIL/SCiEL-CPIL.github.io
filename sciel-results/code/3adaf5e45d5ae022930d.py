"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    import time
    import sys
    sys.setrecursionlimit(100000)
    
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    def find_hamiltonian_cycle(used_edges):
        """Find one Hamiltonian cycle avoiding used_edges. Returns list of vertices or None."""
        # Start from vertex 0, fix direction by going to smallest available neighbor
        start = 0
        visited = [False] * n
        path = [start]
        visited[start] = True
        
        # Get available neighbors of start
        avail_start = [v for v in sorted(adj[start]) if edge_key(start, v) not in used_edges]
        if not avail_start:
            return None
        
        # Try each first step
        for first in avail_start:
            visited[first] = True
            path.append(first)
            
            result = _dfs_ham(first, 2, path, visited, used_edges, deadline)
            if result is not None:
                return result
            path.pop()
            visited[first] = False
        
        return None
    
    def _dfs_ham(current, path_len, path, visited, used_edges, deadline):
        if time.time() > deadline:
            return None
        if path_len == n:
            # Check if we can return to start (vertex 0)
            if edge_key(current, 0) not in used_edges:
                return list(path)
            return None
        
        # Heuristic: try neighbors with fewer remaining options first (most constrained)
        neighbors = []
        for v in adj[current]:
            if not visited[v] and edge_key(current, v) not in used_edges:
                # Count remaining unvisited neighbors of v (excluding current)
                rem = sum(1 for w in adj[v] if not visited[w] and w != current and edge_key(v, w) not in used_edges)
                neighbors.append((rem, v))
        
        neighbors.sort()
        
        for _, nxt in neighbors:
            visited[nxt] = True
            path.append(nxt)
            result = _dfs_ham(nxt, path_len + 1, path, visited, used_edges, deadline)
            if result is not None:
                return result
            path.pop()
            visited[nxt] = False
        
        return None
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    used_edges = set()
    cycles = []
    
    for i in range(3):
        if time.time() > deadline:
            break
        cycle = find_hamiltonian_cycle(used_edges)
        if cycle is None:
            break
        # Mark edges as used
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            used_edges.add(edge_key(u, v))
        cycles.append(cycle)
    
    if cycles:
        return cycles
    else:
        return [list(range(n))]
# EVOLVE-BLOCK-END
