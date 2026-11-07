import time

def construct():
    """
    Constructs a complete bipartite graph K_{7,8} on vertices 0..14.
    Partition A: {0, 1, 2, 3, 4, 5, 6} (7 vertices)
    Partition B: {7, 8, 9, 10, 11, 12, 13, 14} (8 vertices)
    
    In K_{7,8}, every vertex in A has degree 8, and every vertex in B has degree 7.
    Any simple path of length 3 (s-u-v-t) in a bipartite graph has endpoints s and t
    in different partitions.
    Thus, deg(s) is either 7 or 8, and deg(t) is the other.
    Therefore, deg(s) != deg(t) for all such paths.
    
    Total edges = 7 * 8 = 56.
    """
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 15
    # Partition A: vertices 0 to 6
    # Partition B: vertices 7 to 14
    
    edges = []
    for u in range(7):
        if time.time() > deadline:
            break
        for v in range(7, 15):
            if time.time() > deadline:
                break
            edges.append([u, v])
            
    # Ensure types are correct
    final_edges = []
    for e in edges:
        if isinstance(e[0], int) and isinstance(e[1], int):
            final_edges.append([int(e[0]), int(e[1])])
            
    return final_edges
