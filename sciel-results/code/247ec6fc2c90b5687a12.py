import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = P['n']
    # Build adjacency list for the full graph
    adj = {i: [] for i in range(n)}
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        
    # Convert edges to a set of frozensets for fast lookup/removal
    # Or just a set of tuples (min, max)
    original_edges = set()
    for u, v in P['edges']:
        original_edges.add((min(u, v), max(u, v)))
        
    def find_ham_cycle(available_edges):
        """
        Finds a Hamiltonian cycle in the graph defined by available_edges.
        Returns a list of vertices representing the cycle, or None if not found.
        """
        # Build adjacency for the subgraph
        sub_adj = {i: [] for i in range(n)}
        for u, v in available_edges:
            sub_adj[u].append(v)
            sub_adj[v].append(u)
            
        # Check if any vertex has degree < 2, if so no Hamiltonian cycle exists
        for i in range(n):
            if len(sub_adj[i]) < 2:
                return None
                
        start = 0
        path = [0]
        used_edges = set()
        
        def dfs(u, path, used_edges):
            if len(path) == n:
                # Check if edge back to start exists
                if (min(u, start), max(u, start)) in available_edges:
                    return path[:]
                return None
            
            # Heuristic: Sort neighbors by their degree in the remaining graph
            # We need to calculate degree of neighbors in the graph formed by (available_edges - used_edges)
            # This is expensive. A simpler heuristic: Warnsdorff's rule on the current subgraph.
            # Let's just use the static degree from sub_adj, or dynamic if cheap.
            
            # Dynamic degree calculation for neighbors
            neighbors = sub_adj[u]
            # Filter out neighbors already in path (except start at the end)
            # Actually, we track used_edges, so we can check availability.
            
            # Sort neighbors by how many unused edges they have to other unvisited nodes?
            # Simplified: Sort by degree in sub_adj (static)
            # Better: Sort by degree in the graph of available edges not yet used.
            
            def get_remaining_degree(v):
                if v == start and len(path) < n - 1:
                    return 0 # Cannot go back to start until the end
                count = 0
                for w in sub_adj[v]:
                    e = (min(v, w), max(v, w))
                    if e not in used_edges and w not in path: # w not in path is a simplification, but path is a simple path
                        # Actually, w can be in path if it's the start? No, simple path.
                        # But we need to ensure the edge is available.
                        count += 1
                return count
            
            # This inner function is called for every neighbor at every step.
            # n=21, depth 21. 21^2 * 21 is manageable.
            
            # Optimization: Precompute or cache? 
            # Let's try sorting by get_remaining_degree
            
            sorted_neighbors = sorted(neighbors, key=get_remaining_degree)
            
            for v in sorted_neighbors:
                if v in path:
                    continue
                e = (min(u, v), max(u, v))
                if e in used_edges:
                    continue
                # Check if v is start and we are not at the last step
                if v == start and len(path) != n - 1:
                    continue
                
                used_edges.add(e)
                path.append(v)
                
                result = dfs(v, path, used_edges)
                if result:
                    return result
                    
                path.pop()
                used_edges.remove(e)
                
            return None
            
        return dfs(start, path, used_edges)

    best_cycles = []
    
    # Try to find 3 disjoint cycles
    # Strategy: Greedy with restarts if it fails
    max_restarts = 200
    restart_count = 0
    
    while restart_count < max_restarts:
        if time.time() > deadline:
            break
            
        current_edges = set(original_edges)
        cycles = []
        
        success = True
        for i in range(3):
            # Find a cycle in current_edges
            # To add randomness, we can shuffle the start node or neighbor order?
            # The DFS above uses a fixed start (0) and deterministic sorting (though dynamic).
            # To explore different solutions, we can randomize the neighbor sorting tie-breaker
            # or the start node.
            
            # Let's modify find_ham_cycle to accept a random seed or shuffle
            # For now, let's just try. If it fails, we restart.
            
            cycle = find_ham_cycle(current_edges)
            if not cycle:
                success = False
                break
            
            cycles.append(cycle)
            # Remove edges of this cycle from current_edges
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in current_edges:
                    current_edges.remove(e)
                else:
                    # Should not happen if cycle is valid
                    success = False
                    break
            if not success:
                break
                
        if success:
            best_cycles = cycles
            break
            
        restart_count += 1
        # Add some randomness to the next attempt if we were to modify the solver
        # Since the solver is deterministic, we need to change the input order or logic.
        # We can shuffle the edges in P['edges']? No, P is fixed.
        # We can shuffle the adjacency list order?
        # Let's modify find_ham_cycle to shuffle neighbors if a seed is provided.
        
    if not best_cycles:
        # Fallback: Return the single cycle from the seed if we found nothing better
        # Or try to find just 1 or 2.
        # The seed returns 1. We should at least return 1 if we can't find 3.
        # Let's try to find 1, then 2, etc.
        current_edges = set(original_edges)
        cycles = []
        for i in range(3):
            if time.time() > deadline:
                break
            cycle = find_ham_cycle(current_edges)
            if not cycle:
                break
            cycles.append(cycle)
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in current_edges:
                    current_edges.remove(e)
        if cycles:
            best_cycles = cycles
        else:
            # Hardcoded fallback from seed
            best_cycles = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
            
    # Convert to lists of ints
    result = [list(c) for c in best_cycles]
    return result

# EVOLVE-BLOCK-END
