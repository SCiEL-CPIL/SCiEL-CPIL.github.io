"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(n, edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Sort neighbors for deterministic ordering
    for u in adj:
        adj[u].sort()
    return adj

def _find_hamiltonian_cycle(adj, n, forbidden_edges, start=0):
    """Find a Hamiltonian cycle avoiding forbidden edges, starting from 'start'."""
    # forbidden_edges is a set of frozenset({u,v})
    path = [start]
    visited = {start}
    
    def dfs(current, path, visited):
        if len(path) == n:
            # Check if we can close the cycle
            if frozenset((current, start)) not in forbidden_edges:
                return list(path)
            return None
        for next_v in adj[current]:
            if next_v in visited:
                continue
            if frozenset((current, next_v)) in forbidden_edges:
                continue
            path.append(next_v)
            visited.add(next_v)
            result = dfs(next_v, path, visited)
            if result is not None:
                return result
            path.pop()
            visited.remove(next_v)
        return None
    
    return dfs(start, path, visited)

def _find_hamiltonian_cycle_with_backtracking(adj, n, forbidden_edges, deadline, node_limit=500000):
    """More robust Hamiltonian cycle finder with better heuristics."""
    start = 0
    path = [start]
    visited = {start}
    nodes_visited = [0]
    
    def dfs(current, path, visited):
        nodes_visited[0] += 1
        if nodes_visited[0] > node_limit:
            return None
        if len(path) == n:
            if frozenset((current, start)) not in forbidden_edges:
                return list(path)
            return None
        # Order neighbors: prefer those with fewer remaining options (MRV heuristic)
        candidates = []
        for next_v in adj[current]:
            if next_v in visited:
                continue
            if frozenset((current, next_v)) in forbidden_edges:
                continue
            # Count available neighbors for next_v (excluding visited and forbidden)
            avail = 0
            for nb in adj[next_v]:
                if nb not in visited and frozenset((next_v, nb)) not in forbidden_edges:
                    avail += 1
            # Also check if next_v can reach start if it's the last vertex
            candidates.append((avail, next_v))
        candidates.sort()
        for _, next_v in candidates:
            path.append(next_v)
            visited.add(next_v)
            result = dfs(next_v, path, visited)
            if result is not None:
                return result
            path.pop()
            visited.remove(next_v)
        return None
    
    return dfs(start, path, visited)

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(n, edges)
    
    forbidden = set()
    cycles = []
    
    for attempt in range(3):
        if time.time() > deadline:
            break
        
        # Try to find a Hamiltonian cycle
        cycle = None
        
        # Try different starting vertices and strategies
        for start in range(n):
            if time.time() > deadline:
                break
            cycle = _find_hamiltonian_cycle_with_backtracking(adj, n, forbidden, deadline, node_limit=200000)
            if cycle is not None:
                break
        
        if cycle is None:
            break
        
        # Verify the cycle
        if len(cycle) != n or len(set(cycle)) != n:
            break
        valid = True
        for i in range(n):
            u = cycle[i]
            v = cycle[(i+1) % n]
            if frozenset((u, v)) in forbidden:
                valid = False
                break
        if not valid:
            break
        
        # Mark edges as used
        for i in range(n):
            u = cycle[i]
            v = cycle[(i+1) % n]
            forbidden.add(frozenset((u, v)))
        
        cycles.append(cycle)
    
    # Ensure we return at least 1 cycle (the trivial one if nothing else works)
    if not cycles:
        # Fallback: return a single Hamiltonian cycle if we can find one
        cycle = _find_hamiltonian_cycle(adj, n, set())
        if cycle is not None:
            cycles = [cycle]
        else:
            # Absolute fallback
            cycles = [list(range(n))]
    
    # Convert to proper Python types
    result = []
    for c in cycles:
        result.append(tuple(int(v) for v in c))
    
    return result

# EVOLVE-BLOCK-END
