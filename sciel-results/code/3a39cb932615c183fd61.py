import time

def construct():
    """
    Constructs a maximal edge-critical diameter-2 graph on 16 vertices.
    
    The optimal graph is the complete bipartite graph K_{8,8}.
    - Vertices are partitioned into two sets A and B of size 8 each.
    - All edges connect a vertex in A to a vertex in B.
    - Total edges: 8 * 8 = 64.
    - Diameter is 2.
    - Edge-critical: Removing any edge (u, v) where u in A and v in B 
      increases the distance between u and v to 3 (since they have no common neighbors).
      This increases the diameter to 3.
    - By Mantel's Theorem, any triangle-free graph on 16 vertices has at most 64 edges.
      Since any edge in a triangle would not be critical (removing it leaves a path of length 2),
      the graph must be triangle-free. Thus 64 is the maximum possible edge count.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    n = 16
    half = n // 2
    
    # Partition vertices into two sets of size 8
    set_a = list(range(half))
    set_b = list(range(half, n))
    
    edges = []
    
    # Generate all edges between A and B
    for u in set_a:
        for v in set_b:
            # Ensure types are strictly Python int
            edges.append([int(u), int(v)])
            
    # The construction is deterministic and fast, no search needed.
    # Return the list of edges
    return edges
