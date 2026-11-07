import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
    
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Build edge index mapping
    edge_set = set()
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        edge_index[key] = i
    
    # Build adjacency
    adj = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Try MILP approach first
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # MILP formulation for max edge-disjoint Hamiltonian cycles
        # Variables: x[e, c] = 1 if edge e is used in cycle c
        # Max C = 3 (max possible: 70 edges / 21 per cycle)
        max_cycles = 3
        num_vars = m * max_cycles  # x[e,c] for e in 0..m-1, c in 0..max_cycles-1
        
        # Objective: maximize sum of all x[e,c] (each cycle uses exactly n edges, so maximize total)
        c_obj = np.ones(num_vars)
        
        # Constraints:
        # 1. Each edge used in at most one cycle: for each edge e, sum_c x[e,c] <= 1
        # 2. Flow conservation: for each vertex v and cycle c, exactly 2 edges incident to v are used
        # 3. Subtour elimination (MTZ or similar) - this is hard for MILP with n=21
        
        # Actually, let's use a different approach: sequential but with better pruning
        # Use MILP for the assignment, then verify cycles
        
        # Simpler: try to find 3 cycles using improved backtracking with better heuristics
        pass
        
    except ImportError:
        pass
    
    # Fallback: improved sequential backtracking with multiple strategies
    def find_hamiltonian_cycle(forbidden_edges, start_vertex=0, time_limit=5.0):
        """Find Hamiltonian cycle avoiding forbidden edges using DFS with pruning."""
        t_start = time.time()
        adj_local = defaultdict(set)
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key not in forbidden_edges:
                adj_local[u].add(v)
                adj_local[v].add(u)
        
        # Check necessary condition: every vertex must have degree >= 2
        for v in range(n):
            if len(adj_local[v]) < 2:
                return None
        
        def dfs(path, visited):
            if time.time() - t_start > time_limit:
                return None
            
            if len(path) == n:
                last = path[-1]
                first = path[0]
                if first in adj_local[last]:
                    return list(path)
                return None
            
            current = path[-1]
            neighbors = list(adj_local[current])
            unvisited = [nb for nb in neighbors if nb not in visited]
            
            if not unvisited:
                return None
            
            # Warnsdorff's heuristic: prefer vertices with fewer unvisited neighbors
            def score(nb):
                return len([x for x in adj_local[nb] if x not in visited])
            
            unvisited.sort(key=score)
            
            for nb in unvisited:
                path.append(nb)
                visited.add(nb)
                result = dfs(path, visited)
                if result is not None:
                    return result
                path.pop()
                visited.remove(nb)
            
            return None
        
        return dfs([start_vertex], {start_vertex})
    
    forbidden = set()
    cycles = []
    
    # Try to find up to 3 cycles
    for cycle_num in range(3):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple starting vertices and random restarts
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Vary starting vertex for diversity
            start_v = attempt % n
            cycle = find_hamiltonian_cycle(forbidden, start_v, time_limit=2.0)
            
            if cycle:
                # Validate
                valid = len(set(cycle)) == n
                if valid:
                    for j in range(n):
                        u = cycle[j]
                        v = cycle[(j + 1) % n]
                        key = (min(u, v), max(u, v))
                        if key not in edge_set or key in forbidden:
                            valid = False
                            break
                
                if valid:
                    cycles.append(cycle)
                    for j in range(n):
                        u = cycle[j]
                        v = cycle[(j + 1) % n]
                        forbidden.add((min(u, v), max(u, v)))
                    found = True
                    break
        
        if not found:
            break
    
    return cycles
