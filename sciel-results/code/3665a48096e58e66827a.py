import time

def construct():
    """
    Constructs the maximum edge-count graph on 16 vertices with diameter 2 
    where every edge is essential.
    
    Logic:
    1. For every edge to be essential for diameter 2, the graph must be triangle-free.
       (If an edge is in a triangle, removing it leaves a path of length 2, so 
        the diameter might not increase above 2, or specifically, the edge is not 
        'essential' in the sense that its removal doesn't force the distance between 
        its endpoints to exceed 2).
    2. To maximize edges in a triangle-free graph on n vertices, we use Turan's theorem,
       which states the complete bipartite graph with balanced partitions is optimal.
    3. For n=16, the optimal graph is K_{8,8} with 64 edges.
    4. K_{8,8} has diameter 2.
    5. Removing any edge from K_{8,8} increases the distance between the endpoints 
       to 3, thus increasing the diameter to 3.
    
    This solution is theoretically optimal.
    """
    n = 16
    # Partition vertices into two sets of size 8
    # Using list comprehension for distinct implementation
    set_a = list(range(8))
    set_b = list(range(8, 16))
    
    edges = []
    # Generate all edges between set_a and set_b
    for u in set_a:
        for v in set_b:
            # Ensure consistent ordering [min, max]
            if u < v:
                edges.append([u, v])
            else:
                edges.append([v, u])
                
    # Return as list of lists of Python ints
    return [[int(e[0]), int(e[1])] for e in edges]
