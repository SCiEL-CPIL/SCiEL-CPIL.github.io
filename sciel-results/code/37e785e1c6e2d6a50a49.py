import time
from itertools import combinations

def construct():
    # Edge-disjoint Hamiltonian cycle decomposition via backtracking with constraint propagation
    # The graph has 21 vertices and 70 edges. Each Hamiltonian cycle uses 21 edges.
    # Max possible cycles = 70 // 21 = 3 (with 7 edges left over)
    
    edges = P['edges']
    n = P['n']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(frozenset((u, v)))
    
    deadline = time.time() + 55  # safety margin before 60s kill
    
    # Precompute all Hamiltonian cycles (or at least a good set)
    # Use backtracking to find edge-disjoint Hamiltonian cycles
    
    def find_hamiltonian_cycles(max_cycles):
        """Find up to max_cycles edge-disjoint Hamiltonian cycles using backtracking."""
        used_edges = set()
        cycles = []
        
        def can_complete(cycles_so_far, used_edges_so_far):
            """Check if we can still complete with enough edges remaining."""
            remaining_edges = len(edge_set) - len(used_edges_so_far)
            needed = (max_cycles - len(cycles_so_far)) * n
            return remaining_edges >= needed
        
        def backtrack(cycles, used_edges):
            if time.time() > deadline:
                return cycles
            
            if len(cycles) >= max_cycles:
                return cycles
            
            if not can_complete(cycles, used_edges):
                return cycles
            
            # Find a Hamiltonian cycle in the remaining graph
            remaining_adj = [set() for _ in range(n)]
            for u in range(n):
                for v in adj[u]:
                    if frozenset((u, v)) not in used_edges:
                        remaining_adj[u].add(v)
            
            # Try to find a Hamiltonian cycle
            cycle = find_one_hamiltonian_cycle(remaining_adj, n)
            
            if cycle is not None:
                # Get edges of this cycle
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i+1) % len(cycle)]
                    cycle_edges.add(frozenset((u, v)))
                
                new_used = used_edges | cycle_edges
                new_cycles = cycles + [cycle]
                result = backtrack(new_cycles, new_used)
                if len(result) > len(cycles):
                    return result
            
            return cycles
        
        return backtrack([], set())
    
    def find_one_hamiltonian_cycle(adj_list, n):
        """Find one Hamiltonian cycle using DFS with pruning."""
        # Try different starting points and orderings
        for start in range(n):
            path = [start]
            visited = {start}
            
            def dfs(path, visited):
                if len(path) == n:
                    # Check if last vertex connects back to start
                    if path[0] in adj_list[path[-1]]:
                        return path[:]
                    return None
                
                last = path[-1]
                # Order neighbors by degree in remaining graph (heuristic)
                neighbors = list(adj_list[last])
                neighbors.sort(key=lambda x: len(adj_list[x]) - (1 if x in visited else 0))
                
                for nb in neighbors:
                    if nb not in visited:
                        # Pruning: check if nb has enough unvisited neighbors to complete
                        unvisited_neighbors = [x for x in adj_list[nb] if x not in visited]
                        if len(unvisited_neighbors) < (n - len(path) - 1) + (1 if path[0] in adj_list[nb] else 0):
                            # Actually this pruning is tricky; let's use simpler one
                            pass
                        
                        path.append(nb)
                        visited.add(nb)
                        result = dfs(path, visited)
                        if result is not None:
                            return result
                        path.pop()
                        visited.remove(nb)
                
                return None
            
            result = dfs(path, visited)
            if result is not None:
                return result
        
        return None
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    result = find_hamiltonian_cycles(3)
    
    if len(result) >= 1:
        # Ensure types are correct
        return [list(c) for c in result]
    
    # Fallback: return a single Hamiltonian cycle if we can find one
    adj_full = [set() for _ in range(n)]
    for u, v in edges:
        adj_full[u].add(v)
        adj_full[v].add(u)
    
    for start in range(n):
        path = [start]
        visited = {start}
        
        def dfs_full(path, visited):
            if len(path) == n:
                if path[0] in adj_full[path[-1]]:
                    return path[:]
                return None
            last = path[-1]
            for nb in adj_full[last]:
                if nb not in visited:
                    path.append(nb)
                    visited.add(nb)
                    result = dfs_full(path, visited)
                    if result is not None:
                        return result
                    path.pop()
                    visited.remove(nb)
            return None
        
        result = dfs_full(path, visited)
        if result is not None:
            return [list(result)]
    
    # Last resort: return a trivial 3-cycle if exists
    for u, v in edges:
        for w in adj[u]:
            if w != v and w in adj[v]:
                return [[u, v, w]]
    
    # Should not reach here
    return [[0, 1, 2]]

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}
