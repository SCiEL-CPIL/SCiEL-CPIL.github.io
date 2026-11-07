import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 55.0  # Safety margin under 60s
    n = P['n']
    
    # Build adjacency and edge set
    edge_set = set()
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    def is_connected(available_edges, start=0):
        """Check if graph formed by available_edges is connected."""
        if not available_edges:
            return n == 1
        adj_temp = {i: set() for i in range(n)}
        for u, v in available_edges:
            adj_temp[u].add(v)
            adj_temp[v].add(u)
        
        visited = set()
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            for nb in adj_temp[node]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        return len(visited) == n
    
    def find_hamiltonian_cycle(available_edges, start, time_limit):
        """Find a Hamiltonian cycle using backtracking with strong pruning."""
        t0 = time.time()
        path = [start]
        visited = {start}
        
        def backtrack(available):
            if time.time() - t0 > time_limit:
                return "timeout"
            
            if len(path) == n:
                last = path[-1]
                if edge_key(last, start) in available:
                    return list(path)
                return None
            
            last = path[-1]
            # Get candidate neighbors
            candidates = []
            for nb in adj[last]:
                if nb not in visited:
                    if edge_key(last, nb) in available:
                        candidates.append(nb)
            
            if not candidates:
                return None
            
            # Sort candidates by degree in remaining graph (most constrained first)
            def cand_degree(nb):
                d = 0
                for u, v in available:
                    if u == nb or v == nb:
                        d += 1
                return d
            
            candidates.sort(key=lambda x: cand_degree(x))
            
            for nb in candidates:
                ek = edge_key(last, nb)
                if ek not in available:
                    continue
                
                new_available = set(available)
                new_available.discard(ek)
                
                # Pruning 1: Degree constraint for unvisited vertices
                valid = True
                unvisited_set = set(range(n)) - visited - {nb}
                
                if unvisited_set:
                    degs = {v: 0 for v in unvisited_set}
                    for u, v in new_available:
                        if u in degs:
                            degs[u] += 1
                        if v in degs:
                            degs[v] += 1
                    for v in unvisited_set:
                        if degs[v] < 2:
                            valid = False
                            break
                
                # Check nb has at least 1 remaining edge
                if valid:
                    nb_deg = 0
                    for u, v in new_available:
                        if u == nb or v == nb:
                            nb_deg += 1
                    if nb_deg < 1:
                        valid = False
                
                # Pruning 2: Connectivity check
                if valid and len(new_available) > 0:
                    if not is_connected(new_available, start):
                        valid = False
                
                if valid:
                    path.append(nb)
                    visited.add(nb)
                    result = backtrack(new_available)
                    if result is not None and result != "timeout":
                        return result
                    path.pop()
                    visited.remove(nb)
            
            return None
        
        result = backtrack(set(available_edges))
        if result == "timeout":
            return None
        return result
    
    # Strategy: Try to find 3 edge-disjoint Hamiltonian cycles
    cycles = []
    available = set(edge_set)
    random.seed(42)
    
    for cycle_idx in range(3):
        if time.time() > deadline:
            break
        
        found = None
        max_restarts = 25
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Shuffle adjacency lists for different search orders
            for v in range(n):
                adj_list = list(adj[v])
                random.shuffle(adj_list)
                adj[v] = set(adj_list)
            
            start = restart % n
            found = find_hamiltonian_cycle(available, start, time_limit=2.0)
            
            if found is not None:
                break
        
        if found is None:
            break
        
        # Remove used edges
        for i in range(len(found)):
            u = found[i]
            v = found[(i + 1) % len(found)]
            available.discard(edge_key(u, v))
        cycles.append(found)
    
    if cycles:
        return [list(c) for c in cycles]
    else:
        available = set(edge_set)
        found = find_hamiltonian_cycle(available, 0, time_limit=5.0)
        if found:
            return [list(found)]
        return [list(range(n))]
# EVOLVE-BLOCK-END
