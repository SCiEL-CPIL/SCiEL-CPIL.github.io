"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 59  # Safety margin within 60s limit
    
    n = P['n']
    half = n // 2
    
    # The optimal graph is K_{n/2, n/2}, which is triangle-free and has diameter 2.
    # Removing any edge (u, v) with u in A, v in B increases d(u, v) to 3, 
    # thus increasing the diameter to 3.
    
    A = list(range(half))
    B = list(range(half, n))
    
    edges = []
    for u in A:
        if time.time() > deadline:
            break
        for v in B:
            if time.time() > deadline:
                break
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
