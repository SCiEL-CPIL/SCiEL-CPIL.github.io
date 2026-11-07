import time

def construct():
    """
    Returns K_{8,8} on vertices 0..15, which has 64 edges.
    This is the maximum possible edge count for a complete bipartite graph on 16 vertices
    that satisfies the edge-essentiality constraint for diameter 2.
    - Diameter is exactly 2.
    - Every edge is essential: removing any edge (u,v) makes the distance between u and v
      become 3 (since there are no common neighbors in the bipartite structure), 
      thus increasing the diameter to 3.
    """
    deadline = time.time() + 85.0
    
    # Partition vertices into two sets of 8
    part_a = [0, 1, 2, 3, 4, 5, 6, 7]
    part_b = [8, 9, 10, 11, 12, 13, 14, 15]
    
    edge_list = []
    for a in part_a:
        if time.time() > deadline:
            break
        for b in part_b:
            if time.time() > deadline:
                break
            edge_list.append([int(a), int(b)])
            
    return edge_list
