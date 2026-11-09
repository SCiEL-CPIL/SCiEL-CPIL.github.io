import time

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph with 8 disjoint copies of K4.
    n = 32.
    Each K4 has alpha = 1.
    Total alpha = 8.
    K4 contains no 5-cycle.
    Disjoint union contains no 5-cycle.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    # Check if we are close to deadline (not expected for this O(1) task)
    if time.time() > deadline:
         pass # Return best found (which is this)
         
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
