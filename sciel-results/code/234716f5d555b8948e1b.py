"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin
    
    edges = []
    
    # Component 1: K3,3 on vertices 0..5
    # Partition 1: 0, 1, 2
    # Partition 2: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Check deadline (unlikely to be needed for static construction)
    if time.time() > deadline:
        return edges

    # Component 2: K3,3 on vertices 6..11
    # Partition 1: 6, 7, 8
    # Partition 2: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
