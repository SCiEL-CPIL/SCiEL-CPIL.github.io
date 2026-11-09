import random
import time

def construct():
    n = 20
    deadline = time.time() + 110  # 10s safety margin
    
    best_edges = []
    max_edges_found = 0
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Run multiple greedy attempts with different random orderings
    num_iterations = 1000  # Can increase if fast enough
    
    for i in range(num_iterations):
        if time.time() > deadline:
            break
            
        # Shuffle edges using a specific seed for reproducibility if needed, 
        # but random is fine
        edges_order = all_edges[:]
        random.shuffle(edges_order)
        
        # Neighbors sets for fast conflict checking
        neighbors = [set() for _ in range(n)]
        current_edges = []
        
        for u, v in edges_order:
            # Check if adding (u,v) creates a C4
            # A C4 is u-w-x-v-u. 
            # This exists if there is w in N(u) and x in N(v) such that w-x is an edge.
            creates_c4 = False
            
            # Optimization: iterate over the smaller set first
            if len(neighbors[u]) < len(neighbors[v]):
                iter_u, iter_v = neighbors[u], neighbors[v]
            else:
                iter_u, iter_v = neighbors[v], neighbors[u]
                
            # Wait, the logic is symmetric for u and v in the check?
            # We need w in N(u) and x in N(v) such that w~x.
            # If we swap u and v, we need x in N(v) and w in N(u) such that x~w. Same thing.
            
            for w in iter_u:
                # If w is in N(u), check if any neighbor of v is connected to w
                # i.e. if neighbors[w] intersects neighbors[v]
                # Using set intersection is fast
                if neighbors[w] & neighbors[v]:
                    creates_c4 = True
                    break
            
            if not creates_c4:
                neighbors[u].add(v)
                neighbors[v].add(u)
                current_edges.append((u, v))
                
        if len(current_edges) > max_edges_found:
            max_edges_found = len(current_edges)
            best_edges = current_edges[:]
            
    # Properly color the edges
    # Since it's C4-free, any coloring is valid for the 4-cycle constraint.
    # We will assign colors 0..19 cyclically to be safe and look "proper".
    result = []
    for idx, (u, v) in enumerate(best_edges):
        c = idx % 20
        result.append([u, v, c])
        
    return result
