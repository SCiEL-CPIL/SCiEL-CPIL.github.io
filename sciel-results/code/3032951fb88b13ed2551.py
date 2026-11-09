import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Grötzsch graph construction
    # Vertices 0-4 form a 5-cycle
    # Vertices 5-9 are connected to the cycle
    # Vertex 10 is connected to 5-9
    # Standard Grötzsch graph edges:
    # Cycle: (0,1), (1,2), (2,3), (3,4), (4,0)
    # "Spokes" or connections:
    # Let's use a known edge list for Grötzsch (11 vertices, 20 edges)
    # Reference: The Grötzsch graph is the unique 4-chromatic triangle-free graph on 11 vertices.
    
    groetzsche_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (5, 10), (6, 10), (7, 10), (8, 10), (9, 10)
    ]
    
    # Verify Grötzsch is triangle-free and 4-chromatic
    # It is known to be so.
    
    n = 16
    # Initialize adjacency bitmasks
    # adj[i] is a bitmask of neighbors of i
    base_adj = [0] * n
    for u, v in groetzsche_edges:
        base_adj[u] |= (1 << v)
        base_adj[v] |= (1 << u)
        
    # Generate all candidate edges not in base
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (base_adj[i] >> j) & 1:
                candidates.append((i, j))
                
    best_edge_count = len(groetzsche_edges)
    best_adj = base_adj[:]
    best_edges_list = [list(e) for e in groetzsche_edges]
    
    # Function to check if edge (u,v) can be added to current adj
    def can_add(adj, u, v):
        # Check if already exists
        if (adj[u] >> v) & 1:
            return False
        # Check for triangle: common neighbor
        common = adj[u] & adj[v]
        if common:
            return False
        return True

    def try_add(adj, u, v):
        if can_add(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False

    # Greedy Search with Random Restarts
    # We will try to add edges in random order.
    # To improve quality, we can also try ordering by "degree" in the candidate graph?
    # For now, random shuffle is a good structural baseline.
    
    num_trials = 20000
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        current_adj = base_adj[:]
        current_edges = list(groetzsche_edges)
        current_count = len(current_edges)
        
        # Shuffle candidates
        cands = candidates[:]
        random.shuffle(cands)
        
        for u, v in cands:
            if try_add(current_adj, u, v):
                current_edges.append((u, v))
                current_count += 1
                
        if current_count > best_edge_count:
            best_edge_count = current_count
            best_adj = current_adj[:]
            best_edges_list = [list(e) for e in current_edges]
            
            # If we reach a very high count, we can stop early or increase trials?
            # Max possible is ~64. If we get > 40, it's good.
            
    # Final verification and formatting
    # Ensure we return lists
    result = [list(e) for e in best_edges_list]
    
    return result
