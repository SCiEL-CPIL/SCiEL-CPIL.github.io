import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _find_hamiltonian_cycle(n, edges_set):
    """
    Finds a Hamiltonian cycle in the graph defined by n vertices and edges_set.
    Returns a list of vertices or None.
    Modifies edges_set (removes edges as it goes, restores on backtrack).
    """
    if len(edges_set) < n:
        return None
        
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
        
    # Check if any vertex has degree < 2
    for i in range(n):
        if len(adj[i]) < 2:
            return None

    # Backtracking search
    start = 0
    path = [start]
    visited = {start}
    
    # Heuristic: visit neighbors with fewer options first
    # We need to be careful with sorting because adj changes.
    # Sorting at each step is O(k log k), which is fine for small k.
    
    def backtrack():
        if len(path) == n:
            last = path[-1]
            if _edge_key(last, start) in edges_set:
                return True
            return False
        
        last = path[-1]
        neighbors = list(adj[last])
        # Sort by degree to prune early
        neighbors.sort(key=lambda x: len(adj[x]))
        
        for v in neighbors:
            if v == start:
                if len(path) == n - 1:
                    if _edge_key(last, start) in edges_set:
                        path.append(start)
                        return True
                continue
            
            if v not in visited:
                ek = _edge_key(last, v)
                if ek in edges_set:
                    # Remove edge
                    edges_set.remove(ek)
                    adj[last].remove(v)
                    adj[v].remove(last)
                    
                    path.append(v)
                    visited.add(v)
                    
                    if backtrack():
                        return True
                    
                    path.pop()
                    visited.remove(v)
                    
                    # Restore edge
                    edges_set.add(ek)
                    adj[last].add(v)
                    adj[v].add(last)
        
        return False

    if backtrack():
        if path[-1] == start:
            path.pop()
        return path
    return None

def construct():
    n = P['n']
    edges_set = set()
    for u, v in P['edges']:
        edges_set.add(_edge_key(u, v))
        
    cycles = []
    
    # Deadline guard
    start_time = time.time()
    deadline = 80.0  # 80 seconds to be safe
    
    # Try to find 3 cycles
    for _ in range(3):
        if time.time() > deadline:
            break
            
        current_edges = set(edges_set)
        cycle = _find_hamiltonian_cycle(n, current_edges)
        if cycle is None:
            break
        
        cycle = [int(v) for v in cycle]
        cycles.append(cycle)
        
        # Remove edges of this cycle from the main edges_set
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek in edges_set:
                edges_set.remove(ek)

    return cycles
