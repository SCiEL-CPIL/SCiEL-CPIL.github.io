import time
import random
import sys

def construct(k, g):
    if k == 4 and g == 10:
        # Try to find a 4-regular graph H with girth >= 5
        # and return its bipartite double cover.
        best_n = None
        best_edges = None
        
        start_time = time.time()
        deadline = start_time + 110  # Leave 10s for fallback
        
        # Search for H
        for n_h in range(22, 100, 2): # n_h should be even? 4-regular graphs exist for any n>=4.
            if time.time() > deadline:
                break
            
            # Try to find a valid H of size n_h
            # Use a simple backtracking or random search
            h_found = _find_girth5_graph(n_h, time_limit=5)
            if h_found:
                # Construct double cover
                n_g = 2 * n_h
                edges_g = []
                for u, v in h_found:
                    # u, v in 0..n_h-1
                    # Edges in G: (2u, 2v+1) and (2u+1, 2v)
                    edges_g.append((2*u, 2*v + 1))
                    edges_g.append((2*u + 1, 2*v))
                
                # Verify girth is at least 10
                if _check_girth(edges_g, n_g, 10):
                    return n_g, edges_g
        
        # Fallback: LUW graph
        return _luw_construct(k, g)

    else:
        # General case: use LUW
        return _luw_construct(k, g)

def _find_girth5_graph(n, time_limit=5):
    # Backtracking search for 4-regular graph with no 3 or 4 cycles
    # ... implementation ...
    pass

def _check_girth(edges, n, g):
    # BFS from each vertex to depth g//2 - 1
    # ... implementation ...
    pass

def _luw_construct(k, g):
    # Original seed code
    pass
