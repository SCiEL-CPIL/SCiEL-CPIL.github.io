import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_edge_count = 0
    best_edges = []
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper function to check if adding edge (u, v) keeps graph K3,3-free
    def can_add(u, v, adj_mask):
        # If already connected, no change
        if (adj_mask[u] >> v) & 1:
            return True
            
        # We need to ensure that adding (u, v) does not create a K3,3.
        # A K3,3 is created if there exist w, x such that:
        # 1. v is a common neighbor of w and x (v in N(w) and v in N(x))
        # 2. The triple {u, w, x} already has at least 2 common neighbors in the current graph.
        #    Because adding v to N(u) makes v a 3rd common neighbor for {u, w, x}.
        
        mask_u = adj_mask[u]
        
        # Iterate over potential w
        for w in range(n):
            if w == u or w == v:
                continue
            # Check if v is a neighbor of w
            if (adj_mask[w] >> v) & 1:
                # Iterate over potential x
                for x in range(w + 1, n):
                    if x == u or x == v:
                        continue
                    # Check if v is a neighbor of x
                    if (adj_mask[x] >> v) & 1:
                        # Check number of common neighbors of u, w, x
                        common = mask_u & adj_mask[w] & adj_mask[x]
                        if common.bit_count() >= 2:
                            return False
        return True

    # Run multiple random greedy constructions
    while time.time() < deadline:
        # Initialize empty graph
        adj_mask = [0] * n
        edges = set()
        
        # Get a shuffled list of edges
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        # Greedy addition
        for u, v in shuffled_edges:
            if can_add(u, v, adj_mask):
                # Add edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.add((u, v))
        
        # Add-back phase: Try to add edges that were not added in the first pass
        # This can happen if the order was bad.
        # We iterate through all edges not in 'edges'
        missing_edges = [e for e in all_edges if e not in edges]
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if can_add(u, v, adj_mask):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.add((u, v))
        
        # Record best
        if len(edges) > best_edge_count:
            best_edge_count = len(edges)
            best_edges = list(edges)
            
            # If we found a very high count, we might want to try to polish it further?
            # For now, just keep restarting.
            
        # If we have a lot of time, we can do more sophisticated local search on the best graph found so far
        # But random restarts are usually very effective for n=20.
        
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    return result
