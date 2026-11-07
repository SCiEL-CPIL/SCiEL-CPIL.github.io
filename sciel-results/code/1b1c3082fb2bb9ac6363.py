"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55  # safety margin

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def find_hamiltonian_cycle(used_edges):
        """Find a Hamiltonian cycle avoiding used edges. Returns list of vertices or None."""
        start = 0
        # Neighbors of start with available edge
        start_neighbors = []
        for nb in sorted(adj[start]):
            if edge_key(start, nb) not in used_edges:
                start_neighbors.append(nb)
        
        if not start_neighbors:
            return None
        
        # Fix first step to smallest neighbor to break reflection symmetry
        first = start_neighbors[0]
        
        def dfs(path, current, visited):
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                # Check closing edge
                if edge_key(current, start) not in used_edges:
                    return list(path)
                return None
            
            # Pruning: check if remaining unvisited vertices can form a path to start
            # Simple degree check: for each unvisited vertex, it needs at least 2 available edges
            # (one to enter, one to exit), except the last one which needs 1 to enter + 1 to start
            unvisited = [v for v in range(n) if v not in visited]
            for v in unvisited:
                avail_deg = 0
                for nb in adj[v]:
                    if edge_key(v, nb) not in used_edges:
                        if nb == start or nb in visited or nb == v:
                            if nb != v:
                                avail_deg += 1
                        else:
                            avail_deg += 1
                # v needs at least 2 available edges to connect in the path
                # But this is a loose bound; skip for speed
                
            # Try neighbors
            for nb in sorted(adj[current]):
                if nb in visited:
                    continue
                ek = edge_key(current, nb)
                if ek in used_edges:
                    continue
                
                # Pruning: nb must be able to reach start eventually
                # Check if start has an available edge to some unvisited vertex (or nb if nb is last)
                if len(path) == n - 1:
                    # nb must connect back to start
                    if edge_key(nb, start) in used_edges:
                        continue
                
                visited.add(nb)
                path.append(nb)
                result = dfs(path, nb, visited)
                if result is not None:
                    return result
                path.pop()
                visited.discard(nb)
            
            return None
        
        visited = {start}
        path = [start]
        result = dfs(path, first, visited)
        
        if result is not None:
            return result
        
        # Try other first neighbors
        for nb in start_neighbors[1:]:
            if time.time() > deadline:
                return None
            visited = {start}
            path = [start]
            result = dfs(path, nb, visited)
            if result is not None:
                return result
        
        return None

    def find_cycles_greedy(max_cycles):
        """Greedily find edge-disjoint Hamiltonian cycles."""
        used = set()
        cycles = []
        for _ in range(max_cycles):
            if time.time() > deadline:
                break
            c = find_hamiltonian_cycle(used)
            if c is None:
                break
            for i in range(len(c)):
                u, v = c[i], c[(i + 1) % len(c)]
                used.add(edge_key(u, v))
            cycles.append(c)
        return cycles

    # Try to find 3, then 2, then 1
    best = []
    for target in [3, 2, 1]:
        if time.time() > deadline:
            break
        cycles = find_cycles_greedy(target)
        if len(cycles) >= target:
            best = cycles
            break
        elif len(cycles) > len(best):
            best = cycles

    # Convert to list of ints
    result = []
    for c in best:
        result.append([int(v) for v in c])

    return result
# EVOLVE-BLOCK-END
