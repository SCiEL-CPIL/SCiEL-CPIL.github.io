"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

import sys
sys.setrecursionlimit(10000)

def construct():
    n = P['n']
    all_edges = set()
    for e in P['edges']:
        all_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in all_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def find_hamiltonian_cycle_greedy(available_edges):
        """Find a Hamiltonian cycle using greedy + backtracking with heuristics."""
        # Build adjacency from available edges
        adj_avail = [[] for _ in range(n)]
        for (u, v) in available_edges:
            adj_avail[u].append(v)
            adj_avail[v].append(u)
        
        # Sort neighbors by degree for better pruning
        for i in range(n):
            adj_avail[i].sort(key=lambda x: len(adj_avail[x]))
        
        start = 0
        path = [start]
        visited = {start}
        
        def dfs(current, path, visited):
            if len(path) == n:
                if (min(current, start), max(current, start)) in available_edges:
                    return path[:]
                return None
            
            for neighbor in adj_avail[current]:
                if neighbor not in visited:
                    edge = (min(current, neighbor), max(current, neighbor))
                    if edge in available_edges:
                        visited.add(neighbor)
                        path.append(neighbor)
                        result = dfs(neighbor, path, visited)
                        if result is not None:
                            return result
                        path.pop()
                        visited.remove(neighbor)
            return None
        
        result = dfs(start, path, visited)
        if result is not None:
            return result
        return None
    
    def find_hamiltonian_cycle_multi_start(available_edges):
        """Try multiple starting vertices."""
        # Build adjacency from available edges
        adj_avail = [[] for _ in range(n)]
        for (u, v) in available_edges:
            adj_avail[u].append(v)
            adj_avail[v].append(u)
        
        for i in range(n):
            adj_avail[i].sort(key=lambda x: len(adj_avail[x]))
        
        for start in range(n):
            path = [start]
            visited = {start}
            
            def dfs(current, path, visited, start):
                if len(path) == n:
                    if (min(current, start), max(current, start)) in available_edges:
                        return path[:]
                    return None
                
                for neighbor in adj_avail[current]:
                    if neighbor not in visited:
                        edge = (min(current, neighbor), max(current, neighbor))
                        if edge in available_edges:
                            visited.add(neighbor)
                            path.append(neighbor)
                            result = dfs(neighbor, path, visited, start)
                            if result is not None:
                                return result
                            path.pop()
                            visited.remove(neighbor)
                return None
            
            result = dfs(start, path, visited, start)
            if result is not None:
                return result
        return None
    
    # Find cycles iteratively
    cycles = []
    remaining_edges = set(all_edges)
    
    for attempt in range(3):
        found = False
        # Try greedy first
        cycle = find_hamiltonian_cycle_greedy(remaining_edges)
        if cycle is not None:
            cycles.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
            found = True
            continue
        
        # Try multi-start
        cycle = find_hamiltonian_cycle_multi_start(remaining_edges)
        if cycle is not None:
            cycles.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
            found = True
            continue
        
        if not found:
            break
    
    if not cycles:
        # Fallback: try with full edge set
        cycle = find_hamiltonian_cycle_greedy(set(all_edges))
        if cycle is None:
            cycle = find_hamiltonian_cycle_multi_start(set(all_edges))
        if cycle is not None:
            return [cycle]
        return [list(range(n))]
    
    # Ensure all cycles are lists of Python ints
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    return result
# EVOLVE-BLOCK-END
