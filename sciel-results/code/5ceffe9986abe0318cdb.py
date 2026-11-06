import time
import random
from itertools import combinations

def generate_paley(q):
    """Generate the Paley graph on q vertices."""
    if q % 4 != 1:
        return None
    squares = set()
    for i in range(1, q):
        squares.add((i * i) % q)
    
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                edges.append([i, j])
    return {'n': q, 'edges': edges}

def build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def has_k4(adj, n):
    """Check if graph contains a K4."""
    for u in range(n):
        # Get neighbors of u
        neighbors = list(adj[u])
        # Check all pairs of neighbors
        for i in range(len(neighbors)):
            v = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if w in adj[v]:
                    # u, v, w form a triangle
                    # Check if there is a common neighbor of v and w that is also neighbor of u?
                    # Actually, we just need to check if any vertex x is connected to u, v, w.
                    # Since u, v, w are connected, we check intersection of adj[v] and adj[w]
                    # and see if any element is in adj[u].
                    # But we already know v, w are in adj[u].
                    # So we need a 4th vertex x such that x is connected to u, v, w.
                    # x must be in adj[u] \ {v, w}
                    # and x must be in adj[v]
                    # and x must be in adj[w]
                    common_vw = adj[v].intersection(adj[w])
                    # Check if any x in common_vw is also in adj[u]
                    # We know v, w are in adj[u]. We need x != v, w.
                    for x in common_vw:
                        if x in adj[u] and x != v and x != w:
                            return True
    return False

def has_independent_set(adj, n, k):
    """Check if graph contains an independent set of size k."""
    # Use a simple recursive search with pruning for small k
    # Or brute force if n is small enough
    if n > 30 and k > 5:
        # Heuristic: check random subsets? No, need exact.
        # For n=29, k=6, brute force is C(29,6) ~ 475k. Fast.
        pass
        
    def is_independent(vertices):
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if vertices[j] in adj[vertices[i]]:
                    return False
        return True

    # Brute force for n <= 35, k <= 6
    # C(35, 6) is ~ 1.6M. Might be slow in Python if not optimized.
    # But we only do this once or twice.
    for comb in combinations(range(n), k):
        if is_independent(comb):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Try Paley graphs
    # P17 is known valid
    # P29 is candidate
    
    best_graph = None
    best_n = 0
    
    # Check P17
    g17 = generate_paley(17)
    if g17:
        adj17 = build_adjacency(17, g17['edges'])
        if not has_k4(adj17, 17) and not has_independent_set(adj17, 17, 6):
            best_graph = g17
            best_n = 17
    
    # Check P29
    g29 = generate_paley(29)
    if g29:
        adj29 = build_adjacency(29, g29['edges'])
        if not has_k4(adj29, 29) and not has_independent_set(adj29, 29, 6):
            best_graph = g29
            best_n = 29
        else:
            # P29 invalid, stick with P17 for now
            pass
            
    # Try to extend best_graph
    if best_graph is None:
        return {'n': 1, 'edges': []} # Fallback
        
    current_n = best_n
    current_edges = list(best_graph['edges'])
    current_adj = build_adjacency(current_n, current_edges)
    
    # Try to add vertices one by one
    # We will try to add vertices up to n=35 or time limit
    target_n = 35
    
    while current_n < target_n and time.time() < deadline:
        new_v = current_n
        # Try to find a valid adjacency for new_v
        # Strategy: Randomly assign edges, then repair?
        # Or: Start with no edges, add edges to break independent sets?
        
        # Let's try a few random configurations for the new vertex
        found_valid = False
        attempts = 0
        max_attempts = 50
        
        while not found_valid and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            # Random adjacency
            new_adj = set()
            for u in range(current_n):
                if random.random() < 0.5:
                    new_adj.add(u)
            
            # Check K4 constraint:
            # The subgraph induced by new_adj must be triangle-free.
            # If it contains a triangle, we have a K4 with new_v.
            valid_k4 = True
            neighbors_list = list(new_adj)
            for i in range(len(neighbors_list)):
                v1 = neighbors_list[i]
                for j in range(i + 1, len(neighbors_list)):
                    v2 = neighbors_list[j]
                    if v2 in current_adj[v1]:
                        # v1, v2 connected. Check if they share a neighbor in new_adj?
                        # No, just need to check if new_adj contains a triangle.
                        # Triangle in new_adj means 3 vertices in new_adj all connected.
                        # We check pairs. If v1-v2 is edge, we need to check if there is a v3 in new_adj
                        # such that v1-v3 and v2-v3 are edges.
                        for k in range(j + 1, len(neighbors_list)):
                            v3 = neighbors_list[k]
                            if v3 in current_adj[v1] and v3 in current_adj[v2]:
                                valid_k4 = False
                                break
                        if not valid_k4:
                            break
                if not valid_k4:
                    break
            
            if not valid_k4:
                continue
            
            # Check Independent Set constraint:
            # The set of non-neighbors (current_n vertices - new_adj) must not contain an independent set of size 5.
            # Because if it does, adding new_v (which is not connected to them) creates an independent set of size 6.
            non_neighbors = set(range(current_n)) - new_adj
            # Check if G[non_neighbors] has independent set of size 5
            # We can use a helper function
            if has_independent_set_on_subset(current_adj, non_neighbors, 5):
                continue
            
            # If both valid, accept
            found_valid = True
            # Add edges
            for u in new_adj:
                current_edges.append([new_v, u])
                current_adj[u].add(new_v)
            current_adj.append(set(new_adj))
            current_n += 1
            break
            
        if not found_valid:
            # Try a more structured approach? Or give up on this vertex
            # Maybe try adding vertex with specific degree?
            # For now, break if we can't find one easily
            break
            
    return {'n': current_n, 'edges': current_edges}

def has_independent_set_on_subset(adj, vertices, k):
    """Check if induced subgraph on vertices has independent set of size k."""
    if len(vertices) < k:
        return False
    vertices = list(vertices)
    
    def is_ind(v_list):
        for i in range(len(v_list)):
            for j in range(i + 1, len(v_list)):
                if v_list[j] in adj[v_list[i]]:
                    return False
        return True
        
    # Brute force for small k
    for comb in combinations(vertices, k):
        if is_ind(comb):
            return True
    return False
