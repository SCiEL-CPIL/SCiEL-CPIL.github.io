"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency with edge IDs
    adj = {i: [] for i in range(n)}
    edge_set = set()
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        edge_set.add((min(u,v), max(u,v)))
    
    total_edges = len(edges)
    deadline = time.time() + 75  # 75s safety margin (90s limit)
    
    # Precompute neighbor lists for faster access
    neighbors = [adj[i] for i in range(n)]
    
    def find_ham_cycle(used_edges, start=0, time_limit=None):
        """Find a Hamiltonian cycle avoiding used_edges, starting from 'start'."""
        if time_limit is None:
            time_limit = deadline
        
        visited = [False] * n
        visited[start] = True
        path = [start]
        path_edges = []
        
        # Count available edges from each vertex (excluding used)
        # This helps with pruning: if a vertex has 0 available edges and isn't visited, fail
        # But we check dynamically during search
        
        def dfs(current, depth):
            if time.time() > time_limit:
                return None
            if depth == n:
                # Check if edge back to start exists and is unused
                for (nb, eid) in neighbors[current]:
                    if nb == start and eid not in used_edges:
                        return path + [start], path_edges + [eid]
                return None
            
            # Try neighbors - order by fewest remaining options first (MRV heuristic)
            # Get unvisited neighbors with unused edges
            candidates = []
            for (nb, eid) in neighbors[current]:
                if nb == start:
                    continue
                if not visited[nb] and eid not in used_edges:
                    candidates.append((nb, eid))
            
            if not candidates:
                return None
            
            # Sort candidates: prefer vertices with fewer available unvisited neighbors
            # This is a simple degree-based heuristic
            candidates.sort(key=lambda x: sum(1 for (nb2, eid2) in neighbors[x[0]] 
                                             if not visited[nb2] and eid2 not in used_edges and nb2 != start))
            
            for (nb, eid) in candidates:
                if time.time() > time_limit:
                    return None
                visited[nb] = True
                path.append(nb)
                path_edges.append(eid)
                
                result = dfs(nb, depth + 1)
                if result is not None:
                    return result
                
                path.pop()
                path_edges.pop()
                visited[nb] = False
            
            return None
        
        return dfs(start, 1)
    
    def try_find_cycles(used_edges, cycles, max_cycles=3):
        """Try to find up to max_cycles edge-disjoint Hamiltonian cycles."""
        for cycle_num in range(max_cycles):
            if time.time() > deadline:
                break
            
            found = None
            # Try different starting vertices
            starts = list(range(n))
            random.shuffle(starts)
            
            for start in starts:
                if time.time() > deadline:
                    break
                result = find_ham_cycle(used_edges, start)
                if result is not None:
                    found = result
                    break
            
            if found is None:
                break
            
            cycle_path, cycle_eids = found
            cycle = cycle_path[:-1]
            cycles.append(cycle)
            used_edges.update(cycle_eids)
        
        return cycles
    
    # Strategy 1: Deterministic attempt with specific start ordering
    cycles1 = []
    used1 = set()
    for start in [0, 1, 2, 3, 4]:
        if time.time() > deadline:
            break
        result = find_ham_cycle(used1, start)
        if result is not None:
            cycle_path, cycle_eids = result
            cycles1.append(cycle_path[:-1])
            used1.update(cycle_eids)
    
    # Strategy 2: Randomized restarts if Strategy 1 didn't find 3 cycles
    best_cycles = cycles1
    if len(best_cycles) < 3 and time.time() < deadline - 5:
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            used2 = set()
            cycles2 = []
            starts = list(range(n))
            random.shuffle(starts)
            for start in starts:
                if time.time() > deadline - 5:
                    break
                result = find_ham_cycle(used2, start)
                if result is not None:
                    cycle_path, cycle_eids = result
                    cycles2.append(cycle_path[:-1])
                    used2.update(cycle_eids)
                    if len(cycles2) >= 3:
                        break
            if len(cycles2) > len(best_cycles):
                best_cycles = cycles2
                if len(best_cycles) >= 3:
                    break
    
    # Ensure at least one valid cycle
    if not best_cycles:
        # Fallback: try to find just one cycle
        for start in range(n):
            result = find_ham_cycle(set(), start)
            if result is not None:
                best_cycles = [result[0][:-1]]
                break
        if not best_cycles:
            best_cycles = [list(range(n))]
    
    # Validate and convert to proper types
    validated = []
    for cycle in best_cycles:
        # Ensure all elements are Python ints
        cycle = [int(v) for v in cycle]
        # Ensure it's a list
        if isinstance(cycle, (list, tuple)):
            validated.append(list(cycle))
    
    return validated
# EVOLVE-BLOCK-END
