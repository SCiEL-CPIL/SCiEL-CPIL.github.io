"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, used_edges, start=0):
    """Find a Hamiltonian cycle avoiding used edges, starting at 'start'."""
    n = len(adj)
    # available neighbors for each vertex (excluding used edges)
    avail = [set() for _ in range(n)]
    for v in range(n):
        for nb in adj[v]:
            if (v, nb) not in used_edges and (nb, v) not in used_edges:
                avail[v].add(nb)
    
    # Quick check: every vertex must have at least 2 available neighbors
    for v in range(n):
        if len(avail[v]) < 2:
            return None
    
    # DFS with backtracking to find Hamiltonian cycle
    path = [start]
    visited = {start}
    
    def dfs():
        if len(path) == n:
            # Check if last vertex connects back to start
            if start in avail[path[-1]]:
                return True
            return False
        
        current = path[-1]
        for nb in sorted(avail[current]):
            if nb not in visited:
                # Pruning: check if unvisited vertices can still form a path
                # Simple pruning: if nb is the only option for some unvisited vertex, be careful
                path.append(nb)
                visited.add(nb)
                if dfs():
                    return True
                path.pop()
                visited.discard(nb)
        return False
    
    if dfs():
        return list(path)
    return None

def construct():
    deadline = time.time() + 85  # 85s limit, 5s safety margin
    adj = _build_adjacency()
    n = P['n']
    
    cycles = []
    used_edges = set()
    
    # Try to find multiple edge-disjoint Hamiltonian cycles
    max_cycles = 10
    attempts = 0
    
    while len(cycles) < max_cycles and time.time() < deadline:
        attempts += 1
        cycle = _find_hamiltonian_cycle(adj, used_edges)
        if cycle is None:
            break
        
        # Convert cycle to edges
        cycle_edges = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = (min(u, v), max(u, v))
            cycle_edges.add(e)
        
        # Check no overlap with used edges
        if cycle_edges & used_edges:
            break
        
        used_edges.update(cycle_edges)
        cycles.append(cycle)
    
    # Ensure we return at least one valid cycle
    if not cycles:
        # Fallback: return the simple path cycle
        cycles = [list(range(n))]
    
    # Convert to proper types
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
