import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = P['n']
    # Convert edges to a set of frozensets for O(1) lookup
    initial_edges = set()
    for u, v in P['edges']:
        initial_edges.add(frozenset((u, v)))
    
    # Adjacency list for quick neighbor access
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    available_edges = set(initial_edges)
    cycles = []
    
    def find_hamiltonian_cycle(available_edges, adj, n, deadline):
        """
        Finds a Hamiltonian cycle in the graph defined by available_edges.
        Returns the cycle as a list of vertices, or None if not found.
        """
        if time.time() > deadline:
            return None
            
        # Heuristic: Start from a vertex with minimum degree in the remaining graph
        # to constrain the search space early.
        degrees = {v: 0 for v in range(n)}
        for v in range(n):
            for u in adj[v]:
                if frozenset((v, u)) in available_edges:
                    degrees[v] += 1
        
        # If any vertex has degree < 2, no Hamiltonian cycle exists
        min_deg = min(degrees.values())
        if min_deg < 2:
            return None
            
        # Pick a start vertex with minimum degree (tie-break by index for determinism)
        start_vertex = min(range(n), key=lambda v: (degrees[v], v))
        
        # Backtracking search
        # path[0] is start_vertex
        path = [start_vertex]
        visited = {start_vertex}
        
        # Sort neighbors of start vertex by their degree to try constrained ones first
        def get_neighbors(v):
            neighbors = []
            for u in adj[v]:
                if u not in visited and frozenset((v, u)) in available_edges:
                    # Heuristic: prefer neighbors with lower degree in remaining graph
                    deg_u = 0
                    for w in adj[u]:
                        if frozenset((u, w)) in available_edges:
                            deg_u += 1
                    neighbors.append((deg_u, u))
            neighbors.sort()
            return [u for _, u in neighbors]
        
        def backtrack(path, visited):
            if time.time() > deadline:
                return None
                
            if len(path) == n:
                # Check if the last vertex connects back to the start
                last = path[-1]
                if frozenset((last, start_vertex)) in available_edges:
                    return list(path)
                return None
                
            last = path[-1]
            # Get neighbors that are not visited and connected by an available edge
            neighbors = get_neighbors(last)
            
            # Additional heuristic: if we are at the second to last vertex,
            # the next vertex MUST connect to start_vertex
            if len(path) == n - 1:
                for u in neighbors:
                    if frozenset((u, start_vertex)) in available_edges:
                        return path + [u]
                return None
            
            for u in neighbors:
                # Try adding u to the path
                visited.add(u)
                path.append(u)
                
                # Check if this choice leads to a dead end immediately?
                # Simple check: does u have any other available neighbors besides last and start?
                # If len(path) < n-1, u needs at least one more neighbor (besides last) to continue.
                # If len(path) == n-1, u needs to connect to start.
                
                res = backtrack(path, visited)
                if res is not None:
                    return res
                    
                # Undo
                path.pop()
                visited.remove(u)
                
            return None
            
        return backtrack(path, visited)

    # Main loop: find cycles one by one
    while True:
        if time.time() > deadline:
            break
            
        # Check if enough edges remain for a Hamiltonian cycle
        if len(available_edges) < n:
            break
            
        cycle = find_hamiltonian_cycle(available_edges, adj, n, deadline)
        
        if cycle is None:
            break
            
        # Remove edges of the found cycle from available_edges
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = frozenset((u, v))
            if edge in available_edges:
                available_edges.remove(edge)
            else:
                # This should not happen if the search was correct
                break
                
        cycles.append(cycle)
        
        # If we found 3 cycles, we have reached the theoretical maximum (63 edges / 21 per cycle)
        if len(cycles) >= 3:
            break
            
    return cycles

# EVOLVE-BLOCK-END
