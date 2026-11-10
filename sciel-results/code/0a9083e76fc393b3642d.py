"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

# Precompute adjacency list
N = P['n']
adj = {i: [] for i in range(N)}
for u, v in P['edges']:
    adj[u].append(v)
    adj[v].append(u)

def get_edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)

def find_hamiltonian_cycle(adj, forbidden_edges, start_vertex=0, neighbor_order_offset=0):
    """
    Finds a Hamiltonian cycle using backtracking.
    forbidden_edges: set of edge tuples that cannot be used.
    neighbor_order_offset: used to vary the search order for different attempts.
    """
    n = len(adj)
    visited = [False] * n
    path = [start_vertex]
    visited[start_vertex] = True
    used_edges = set(forbidden_edges)
    
    # Store the original adjacency to allow reordering
    orig_adj = {k: list(v) for k, v in adj.items()}
    
    def dfs(v):
        if len(path) == n:
            # Check if closing edge exists
            edge = get_edge_key(v, start_vertex)
            if edge not in used_edges and edge in set(get_edge_key(u, w) for u in orig_adj[v] for w in [start_vertex] if u != w):
                # Verify edge is actually in graph
                if start_vertex in orig_adj[v]:
                    return True
            return False
        
        # Get neighbors, ordered by offset
        neighbors = orig_adj[v]
        # Rotate neighbors based on offset to explore different branches
        if neighbor_order_offset > 0:
            neighbors = neighbors[neighbor_order_offset % len(neighbors):] + neighbors[:neighbor_order_offset % len(neighbors)]
            
        for u in neighbors:
            if not visited[u]:
                edge = get_edge_key(v, u)
                if edge not in used_edges:
                    # Heuristic: prefer vertices with fewer available options?
                    # Simple: just try
                    visited[u] = True
                    path.append(u)
                    used_edges.add(edge)
                    
                    if dfs(u):
                        return True
                    
                    visited[u] = False
                    path.pop()
                    used_edges.remove(edge)
        
        return False

    if dfs(start_vertex):
        return path
    return None

def is_hamiltonian_cycle_in_remaining(remaining_edges, n):
    """
    Checks if the set of remaining edges forms a single Hamiltonian cycle.
    remaining_edges: set of (u, v) tuples.
    """
    if len(remaining_edges) != n:
        return None
    
    # Build adjacency from remaining edges
    rem_adj = {i: [] for i in range(n)}
    for u, v in remaining_edges:
        rem_adj[u].append(v)
        rem_adj[v].append(u)
        
    # Check if all vertices have degree 2
    for i in range(n):
        if len(rem_adj[i]) != 2:
            return None
            
    # Check if it's a single cycle
    visited = [False] * n
    start = 0
    curr = start
    path = [curr]
    visited[curr] = True
    next_v = rem_adj[curr][0]
    
    while next_v != start:
        if visited[next_v]:
            # Cycle closed before visiting all nodes?
            if len(path) == n:
                return path
            else:
                return None
        visited[next_v] = True
        path.append(next_v)
        curr = next_v
        # Get next neighbor
        next_v = rem_adj[curr][0] if rem_adj[curr][1] == path[-2] else rem_adj[curr][1]
        
    if len(path) == n:
        return path
    return None

def construct():
    start_time = time.time()
    deadline = 55.0 # Leave 5s buffer
    
    all_edges = set(get_edge_key(u, v) for u, v in P['edges'])
    best_result = []
    
    # First, find one cycle to ensure we don't return empty
    # Use a deterministic first pass
    c1 = find_hamiltonian_cycle(adj, set(), 0, 0)
    if c1:
        best_result = [c1]
    
    if not best_result:
        return []
        
    # Try to find a second and third cycle
    # We will try several variations of C1 and C2
    
    # To find different C1s, we can vary the neighbor order offset
    # Or just try to find C2 for the first C1, then try to find C3.
    # If C3 fails, try a different C2 for the same C1.
    # If that fails, try a different C1.
    
    attempts = 0
    max_attempts = 50
    
    while time.time() - start_time < deadline and attempts < max_attempts:
        attempts += 1
        
        # Determine C1
        # For the first few attempts, we might stick with the found C1.
        # Then we try to find new C1s.
        if attempts == 1:
            current_c1 = c1
        else:
            # Try to find a different C1
            # We can use the offset to change the search path
            offset = attempts * 3
            current_c1 = find_hamiltonian_cycle(adj, set(), 0, offset)
            if not current_c1:
                continue
            # Avoid duplicates? Simple check: if same as best, skip?
            # Just proceed, it might lead to a different C2.
            
        # Find C2 in G \ C1
        edges_c1 = set(get_edge_key(path[i], path[(i+1)%len(path)]) for i in range(len(path)))
        forbidden_for_c2 = edges_c1
        
        c2 = find_hamiltonian_cycle(adj, forbidden_for_c2, 0, attempts * 7)
        
        if c2:
            edges_c2 = set(get_edge_key(path[i], path[(i+1)%len(path)]) for i in range(len(path)))
            
            # Check for C3
            remaining_edges = all_edges - edges_c1 - edges_c2
            c3 = is_hamiltonian_cycle_in_remaining(remaining_edges, N)
            
            if c3:
                return [current_c1, c2, c3]
            else:
                # We have 2 cycles
                if len(best_result) < 2:
                    best_result = [current_c1, c2]
                # Continue searching for 3?
                # We could try to find a different C2 for this C1.
                # Or just let the loop try a different C1.
                # To save time, if we found 2, we can try a few more times to see if 3 is possible.
                # If we don't find 3 in a few tries, we return 2.
                # The loop condition handles this.
                
    return best_result

# EVOLVE-BLOCK-END
