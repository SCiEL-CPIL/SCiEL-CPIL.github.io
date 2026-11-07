"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard (though this construction is O(1))
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    edges = []
    
    # Component 1: K_3,3 on vertices 0,1,2 and 3,4,5
    for u in range(3):
        for v in range(3, 6):
            edges.append([u, v])
            
    # Component 2: K_3,3 on vertices 6,7,8 and 9,10,11
    for u in range(6, 9):
        for v in range(9, 12):
            edges.append([u, v])
            
    # Check time
    if time.time() > deadline:
        return edges
        
    return edges
# EVOLVE-BLOCK-END
