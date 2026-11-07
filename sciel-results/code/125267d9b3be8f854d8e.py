"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Total number of edges
    total_edges = len(edges)
    
    # Each cycle of length L uses L edges.
    # We want to maximize the number of edge-disjoint cycles on the same vertex set (all 21 vertices).
    # Since all cycles must use exactly the same vertex set of size n=21, each cycle is a Hamiltonian cycle.
    # A Hamiltonian cycle on n vertices has exactly n edges.
    # So each cycle uses 21 edges.
    # Total edges available: let's count.
    
    # Max number of cycles = floor(total_edges / n)
    # But we also need them to be edge-disjoint.
    
    # Let's first find one Hamiltonian cycle, then try to find more.
    
    def find_hamiltonian_cycle(used_edges, adj, n, deadline):
        """Find a Hamiltonian cycle avoiding used_edges."""
        # used_edges is a set of frozensets {u,v}
        # We need a cycle visiting all n vertices exactly once.
        
        # Try all starting vertices (but by symmetry, fix start=0 to reduce search)
        # Actually, we should try multiple starts for robustness, but fix to 0 for speed.
        
        def dfs(path, visited, current):
            if time.time() > deadline:
                return None
            if len(path) == n:
                # Check if current connects back to start
                start = path[0]
                if start in adj[current] and frozenset((current, start)) not in used_edges:
                    return path[:]
                return None
            
            for nxt in sorted(adj[current]):
                if nxt not in visited:
                    if frozenset((current, nxt)) not in used_edges:
                        visited.add(nxt)
                        path.append(nxt)
                        result = dfs(path, visited, nxt)
                        if result is not None:
                            return result
                        path.pop()
                        visited.remove(nxt)
            return None
        
        # Fix start vertex to 0
        start = 0
        visited = {start}
        path = [start]
        result = dfs(path, visited, start)
        return result
    
    # Greedy approach: repeatedly find a Hamiltonian cycle, add it to the list, mark its edges as used.
    # To improve, we can try different orderings or use backtracking.
    
    best_cycles = []
    
    # Deadline: 60 seconds total, use 55 seconds for search
    deadline = time.time() + 55.0
    
    # Try multiple strategies:
    # 1. Greedy with fixed start=0
    # 2. Greedy with varying start vertices
    # 3. Randomized restarts
    
    def try_greedy(start_vertex, deadline):
        used_edges = set()
        cycles = []
        while True:
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle(used_edges, adj, n, deadline)
            if cycle is None:
                break
            # Add edges to used set
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(frozenset((u, v)))
            cycles.append(cycle)
        return cycles
    
    # Strategy 1: Greedy with start=0
    cycles1 = try_greedy(0, deadline)
    best_cycles = cycles1
    
    # Strategy 2: Try different start vertices
    for start in range(1, n):
        if time.time() > deadline:
            break
        cycles2 = try_greedy(start, deadline)
        if len(cycles2) > len(best_cycles):
            best_cycles = cycles2
    
    # Strategy 3: Randomized - shuffle adjacency lists
    import random
    for trial in range(100):
        if time.time() > deadline:
            break
        # Shuffle adjacency lists
        adj_shuffled = [list(s) for s in adj]
        for lst in adj_shuffled:
            random.shuffle(lst)
        # Rebuild adj as sets for the search, but use shuffled order
        # Actually, let's just modify the dfs to use shuffled order
        # Simpler: create a new adj with shuffled lists
        adj_copy = [set(lst) for lst in adj_shuffled]
        
        # Redefine find_hamiltonian_cycle with shuffled adj
        def find_ham_cycle_shuffled(used_edges, adj_local, n, deadline):
            def dfs(path, visited, current):
                if time.time() > deadline:
                    return None
                if len(path) == n:
                    start = path[0]
                    if start in adj_local[current] and frozenset((current, start)) not in used_edges:
                        return path[:]
                    return None
                for nxt in adj_local[current]:
                    if nxt not in visited:
                        if frozenset((current, nxt)) not in used_edges:
                            visited.add(nxt)
                            path.append(nxt)
                            result = dfs(path, visited, nxt)
                            if result is not None:
                                return result
                            path.pop()
                            visited.remove(nxt)
                return None
            start = 0
            visited = {start}
            path = [start]
            return dfs(path, visited, start)
        
        used_edges = set()
        cycles3 = []
        while True:
            if time.time() > deadline:
                break
            cycle = find_ham_cycle_shuffled(used_edges, adj_copy, n, deadline)
            if cycle is None:
                break
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(frozenset((u, v)))
            cycles3.append(cycle)
        
        if len(cycles3) > len(best_cycles):
            best_cycles = cycles3
    
    # Validate and return
    if not best_cycles:
        # Fallback: return a single cycle if we can find one
        cycle = find_hamiltonian_cycle(set(), adj, n, deadline)
        if cycle is not None:
            best_cycles = [cycle]
        else:
            # Should not happen, but return empty to be safe? No, must be nonempty.
            # Return a trivial cycle? But must be simple cycle on all 21 vertices.
            # This should not occur for this graph.
            return []
    
    # Ensure all cycles use the same vertex set
    vertex_set = set(range(n))
    for cycle in best_cycles:
        if set(cycle) != vertex_set:
            best_cycles = []
            break
    
    if not best_cycles:
        # Last resort
        cycle = find_hamiltonian_cycle(set(), adj, n, deadline)
        if cycle is not None:
            best_cycles = [cycle]
    
    # Convert to list of lists of ints
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
