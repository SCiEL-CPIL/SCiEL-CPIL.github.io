"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline to be safe within 90s
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    # Precompute: find a known good set of Hamiltonian cycles
    # The graph looks like it might decompose into multiple Hamiltonian cycles
    # Let's try to find them systematically
    
    # Strategy: Use backtracking to find edge-disjoint Hamiltonian cycles
    # We'll try different starting patterns and use constraint propagation
    
    def is_valid_cycle(cycle, used_edges):
        """Check if a cycle is valid (all edges exist and not used)"""
        if len(cycle) < 3:
            return False
        # Check all vertices distinct
        if len(set(cycle)) != len(cycle):
            return False
        # Check all edges exist and not used
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            edge = (min(u,v), max(u,v))
            if edge not in edge_set:
                return False
            if edge in used_edges:
                return False
        return True
    
    def get_cycle_edges(cycle):
        """Get the set of edges in a cycle"""
        edges_in_cycle = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            edges_in_cycle.add((min(u,v), max(u,v)))
        return edges_in_cycle
    
    # Try to find Hamiltonian cycles using a heuristic approach
    # Fix vertex 0 at the start to reduce symmetry
    # Use backtracking with pruning
    
    best_cycles = []
    best_count = 0
    
    # Pre-compute some candidate cycles by looking at the structure
    # The graph seems to have a regular pattern. Let me try to identify
    # cycles based on the edge structure.
    
    # Let's try a systematic search with time limit
    # We'll use a randomized restart approach with constraint propagation
    
    import random
    random.seed(42)
    
    def find_hamiltonian_cycles(max_cycles=10, time_limit=None):
        """Find edge-disjoint Hamiltonian cycles using backtracking"""
        if time_limit is None:
            time_limit = deadline - time.time()
        
        used_edges = set()
        cycles = []
        
        # Try to build cycles one by one
        # For each cycle, use backtracking with vertex ordering heuristics
        
        def backtrack_cycle(used_edges_local, cycles_local, depth=0):
            nonlocal best_cycles, best_count
            
            if time.time() > deadline - 5:
                return False
            
            if depth >= max_cycles:
                if len(cycles_local) > len(best_cycles):
                    best_cycles = list(cycles_local)
                    best_count = len(cycles_local)
                return True
            
            # Try to find a Hamiltonian cycle using remaining edges
            # Use a greedy + backtracking approach
            
            # Order vertices by degree in remaining graph for better pruning
            remaining_adj = [set() for _ in range(n)]
            for u in range(n):
                for v in adj[u]:
                    edge = (min(u,v), max(u,v))
                    if edge not in used_edges_local:
                        remaining_adj[u].add(v)
            
            # Check if all vertices have degree >= 2 (necessary for Hamiltonian cycle)
            for u in range(n):
                if len(remaining_adj[u]) < 2:
                    return False
            
            # Try to find a Hamiltonian cycle starting from vertex 0
            # Use backtracking
            def find_cycle_from(start, path, visited, edges_used):
                if time.time() > deadline - 5:
                    return None
                
                if len(path) == n:
                    # Check if we can close the cycle
                    last = path[-1]
                    edge = (min(last, start), max(last, start))
                    if edge not in edges_used and edge in edge_set:
                        return list(path)
                    return None
                
                current = path[-1]
                # Try neighbors in order of fewest remaining options (MRV heuristic)
                neighbors = sorted(remaining_adj[current], key=lambda x: len(remaining_adj[x]))
                
                for next_v in neighbors:
                    if next_v not in visited:
                        edge = (min(current, next_v), max(current, next_v))
                        if edge not in edges_used and edge in edge_set:
                            new_path = path + [next_v]
                            new_visited = visited | {next_v}
                            new_edges = edges_used | {edge}
                            result = find_cycle_from(start, new_path, new_visited, new_edges)
                            if result is not None:
                                return result
                
                return None
            
            # Try different orderings of vertices to find cycles
            # Start with vertex 0 fixed
            result = find_cycle_from(0, [0], {0}, set())
            
            if result is not None:
                cycle_edges = get_cycle_edges(result)
                new_used = used_edges_local | cycle_edges
                new_cycles = cycles_local + [result]
                if backtrack_cycle(new_used, new_cycles, depth + 1):
                    return True
            
            return False
        
        backtrack_cycle(used_edges, cycles, 0)
        return best_cycles
    
    # Run the search
    result = find_hamiltonian_cycles(max_cycles=10)
    
    if result:
        # Ensure all are lists of Python ints
        return [list(cycle) for cycle in result]
    else:
        # Fallback: return a single Hamiltonian cycle if we can find one
        # Try a simple greedy approach
        def greedy_hamiltonian():
            # Try to find any Hamiltonian cycle
            def find_cycle(start, path, visited):
                if len(path) == n:
                    last = path[-1]
                    if (min(last, start), max(last, start)) in edge_set:
                        return list(path)
                    return None
                
                current = path[-1]
                for next_v in sorted(adj[current]):
                    if next_v not in visited:
                        new_path = path + [next_v]
                        new_visited = visited | {next_v}
                        result = find_cycle(start, new_path, new_visited)
                        if result is not None:
                            return result
                return None
            
            return find_cycle(0, [0], {0})
        
        cycle = greedy_hamiltonian()
        if cycle:
            return [list(cycle)]
        else:
            # Last resort: return a trivial valid cycle if possible
            # Find any triangle
            for i in range(n):
                for j in adj[i]:
                    if j > i:
                        for k in adj[j]:
                            if k > j and k in adj[i]:
                                return [[i, j, k]]
            # Should not reach here for this graph
            return [[0, 1, 2]]

# EVOLVE-BLOCK-END
