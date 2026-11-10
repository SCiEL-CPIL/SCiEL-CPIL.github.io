import time
import sys
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    # Build adjacency list as sets of edges for quick removal
    remaining_edges = set()
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        if u > v: u, v = v, u
        remaining_edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    cycles_found = []
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    def find_hamiltonian_cycle(start_vertex=0):
        # Returns a list of vertices representing a Hamiltonian cycle, or None
        # Uses backtracking with Warnsdorff's rule and random tie-breaking
        path = [start_vertex]
        visited = {start_vertex}
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                last = path[-1]
                first = path[0]
                edge_key = (first, last) if first < last else (last, first)
                if edge_key in remaining_edges:
                    return path[:]
                return None
            
            curr = path[-1]
            # Heuristic: Sort neighbors by degree in remaining graph (least constraining value)
            # Add random tie-breaking to escape local minima
            neighbors = list(adj[curr])
            # Filter to only valid edges
            valid_neighbors = []
            for nxt in neighbors:
                if nxt not in visited:
                    edge_key = (curr, nxt) if curr < nxt else (nxt, curr)
                    if edge_key in remaining_edges:
                        valid_neighbors.append(nxt)
            
            if not valid_neighbors:
                return None

            # Sort by degree, with random tie-breaker
            valid_neighbors.sort(key=lambda x: (len(adj[x]), random.random()))
            
            for nxt in valid_neighbors:
                path.append(nxt)
                visited.add(nxt)
                
                # Pruning: Check if any unvisited vertex is isolated from the rest
                dead_end = False
                unvisited = [v for v in range(n) if v not in visited]
                if len(unvisited) > 1:
                    for v in unvisited:
                        has_unvisited_neighbor = False
                        for u in adj[v]:
                            if u in unvisited:
                                has_unvisited_neighbor = True
                                break
                        if not has_unvisited_neighbor:
                            dead_end = True
                            break
                
                if not dead_end:
                    res = backtrack()
                    if res is not None:
                        return res
                
                path.pop()
                visited.remove(nxt)
            
            return None
        
        return backtrack()

    # Try to find 3 edge-disjoint Hamiltonian cycles
    # Use multiple starting vertices for robustness
    start_vertices = list(range(n))
    random.shuffle(start_vertices)
    
    while len(cycles_found) < 3:
        if time.time() > deadline:
            break
        
        found = False
        for sv in start_vertices:
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle(sv)
            if cycle is not None:
                # Remove edges from cycle
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge_key = (u, v) if u < v else (v, u)
                    if edge_key in remaining_edges:
                        remaining_edges.remove(edge_key)
                        adj[u].remove(v)
                        adj[v].remove(u)
                    else:
                        valid = False
                        break
                
                if valid:
                    cycles_found.append(cycle)
                    found = True
                    break
        
        if not found:
            break

    if not cycles_found:
        # Fallback: try a simple deterministic approach
        return [list(range(n))]
        
    # Ensure types are correct
    result = []
    for c in cycles_found:
        result.append([int(v) for v in c])
        
    return result
# EVOLVE-BLOCK-END
