"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(adj, start, deadline):
    """Find a Hamiltonian cycle starting from 'start' using Warnsdorff-guided DFS."""
    n = P['n']
    
    def dfs(path, visited, adj):
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check if we can close the cycle
            if path[0] in adj[path[-1]]:
                return path[:]
            return None
        
        current = path[-1]
        # Get unvisited neighbors
        candidates = [v for v in adj[current] if v not in visited]
        
        if not candidates:
            return None
        
        # Warnsdorff: prioritize vertices with fewer onward options
        def warnsdorff_score(v):
            return len([w for w in adj[v] if w not in visited and w != path[0]])
        
        candidates.sort(key=warnsdorff_score)
        
        for v in candidates:
            path.append(v)
            visited.add(v)
            result = dfs(path, visited, adj)
            if result is not None:
                return result
            path.pop()
            visited.remove(v)
        
        return None
    
    visited = {start}
    path = [start]
    return dfs(path, visited, adj)

def construct():
    deadline = time.time() + 50  # 50 second safety margin
    
    edges = list(P['edges'])
    n = P['n']
    
    # Build initial adjacency
    adj = _build_adj(edges)
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    cycles = []
    used_edges = set()
    
    for attempt in range(3):
        if time.time() > deadline:
            break
        
        # Build adjacency from remaining edges
        remaining_edges = [e for e in edges if (min(e), max(e)) not in used_edges]
        adj = _build_adj(remaining_edges)
        
        # Check if all vertices have degree >= 2 (necessary for Hamiltonian cycle)
        valid = True
        for i in range(n):
            if len(adj[i]) < 2:
                valid = False
                break
        if not valid:
            break
        
        # Try multiple starting vertices and randomizations
        found = False
        for start in range(n):
            if time.time() > deadline:
                break
            cycle = _find_ham_cycle(adj, start, deadline)
            if cycle is not None:
                # Mark edges as used
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    key = (min(u, v), max(u, v))
                    used_edges.add(key)
                cycles.append(cycle)
                found = True
                break
        
        if not found:
            break
    
    # Convert to proper types
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
