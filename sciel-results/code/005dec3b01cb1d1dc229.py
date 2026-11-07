import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 5.0
    
    n = 16
    half = n // 2
    
    # Define the two partitions for the complete bipartite graph K_{8,8}
    # Vertices 0..7 in Partition 1, Vertices 8..15 in Partition 2
    part1 = list(range(half))
    part2 = list(range(half, n))
    
    # Generate all edges between the two partitions
    # This creates the complete bipartite graph K_{8,8}
    # Properties:
    # 1. Triangle-free (bipartite graphs are triangle-free)
    # 2. Diameter 2 (any node in P1 is dist 2 from any other in P1 via P2, dist 1 to P2)
    # 3. Edge-critical: Removing any edge (u,v) with u in P1, v in P2 
    #    breaks the direct connection. Since the graph is bipartite, 
    #    no path of length 2 exists between P1 and P2. 
    #    The shortest path becomes length 3 (u->p2_node->p1_node->v).
    #    Thus diameter increases from 2 to 3.
    
    edges = []
    for u in part1:
        if time.time() > deadline:
            break
        for v in part2:
            edges.append([u, v])
            
    return edges
