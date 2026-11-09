"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = 60.0  # 60s limit, use 55s to be safe
    
    n = P['n']
    base_edges = {tuple(e) for e in P['edges']}
    
    # Hub at 12 (Even)
    # Connect to all Odd vertices
    hub = 12
    added_edges = []
    for v in range(1, n, 2):
        if (hub, v) not in base_edges and (v, hub) not in base_edges:
            added_edges.append([hub, v])
            
    # Verify and return
    if time.time() > deadline:
        return added_edges
        
    return added_edges
# EVOLVE-BLOCK-END
