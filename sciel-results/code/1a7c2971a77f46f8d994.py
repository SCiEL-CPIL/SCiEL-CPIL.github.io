"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # The graph is 6-regular, so max 3 Hamiltonian cycles
    # We aim to find 3.
    
    def find_hamiltonian_cycle(adj_remaining, n, start=0, deadline=None):
        """Find a Hamiltonian cycle in the graph using DFS with Warnsdorff's heuristic."""
        visited = [False] * n
        path = [start]
        visited[start] = True
        
        def backtrack(u):
            if deadline and time.time() > deadline:
                return False
            
            if len(path) == n:
                # Check if we can close the cycle
                if start in adj_remaining[u]:
                    return True
                return False
            
            # Get unvisited neighbors
            candidates = [v for v in adj_remaining[u] if not visited[v]]
            
            if not candidates:
                return False
            
            # Warnsdorff's rule: prefer vertices with fewer unvisited neighbors
            # This helps avoid getting stuck in dead ends
            candidates.sort(key=lambda v: len([w for w in adj_remaining[v] if not visited[w]]))
            
            # Randomize ties to explore different paths
            # Shuffle before sorting to randomize tie-breaking
            random.shuffle(candidates)
            candidates.sort(key=lambda v: len([w for w in adj_remaining[v] if not visited[w]]))
            
            for v in candidates:
                # Pruning: if this is the last step, ensure it can connect to start
                if len(path) == n - 1:
                    if start not in adj_remaining[v]:
                        continue
                
                visited[v] = True
                path.append(v)
                if backtrack(v):
                    return True
                path.pop()
                visited[v] = False
            
            return False
        
        if backtrack(start):
            return path[:]
        return None

    best_cycles = []
    deadline = time.time() + 85  # 5s margin
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    # We will try multiple restarts with randomized neighbor orders
    for attempt in range(200):
        if time.time() > deadline:
            break
            
        # Reset graph
        adj_remaining = {i: set() for i in range(n)}
        for u, v in edges:
            adj_remaining[u].add(v)
            adj_remaining[v].add(u)
        
        cycles = []
        success = True
        
        for _ in range(3):
            # Try to find a cycle
            # Random start vertex to break symmetry further
            start = random.randint(0, n - 1)
            
            cycle = find_hamiltonian_cycle(adj_remaining, n, start, deadline)
            
            if cycle is None:
                # Try a few other starts if the first one fails
                found = False
                for s in range(n):
                    if s == start: continue
                    cycle = find_hamiltonian_cycle(adj_remaining, n, s, deadline)
                    if cycle is not None:
                        found = True
                        break
                if not found:
                    success = False
                    break
            
            # Remove edges used in this cycle
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                adj_remaining[u].discard(v)
                adj_remaining[v].discard(u)
            
            cycles.append(cycle)
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles[:]
            if len(best_cycles) == 3:
                break
        
    # Fallback
    if not best_cycles:
        # The seed cycle [0, 1, 2, ..., 20] is likely valid
        # Check if it's valid in original graph
        valid = True
        for i in range(n):
            u = i
            v = (i + 1) % n
            if v not in adj[u]:
                valid = False
                break
        if valid:
            best_cycles = [[i for i in range(n)]]
    
    return [list(c) for c in best_cycles]
# EVOLVE-BLOCK-END
