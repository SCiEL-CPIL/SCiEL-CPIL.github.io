"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a graph on 16 vertices (0..15) with diameter exactly 2
    where every edge is critical (removing any edge increases diameter > 2
    or disconnects the graph).
    
    The graph used is the complete bipartite graph K_{8,8}.
    Partitions: A = {0..7}, B = {8..15}.
    Edge count: 8 * 8 = 64.
    
    Verification:
    1. Diameter is 2:
       - Vertices in A are distance 1 from B and distance 2 from A (via B).
       - Vertices in B are distance 1 from A and distance 2 from B (via A).
    2. Edge criticality:
       - Consider any edge (u, v) with u in A, v in B.
       - In K_{8,8}, N(u) = B and N(v) = A.
       - N(u) intersection N(v) is empty.
       - Removing (u, v) leaves u and v with no common neighbor.
       - Thus, distance between u and v becomes 3 (u -> w in B -> x in A -> v).
       - The diameter of the resulting graph is at least 3.
       - Thus, the edge is critical.
    """
    n = 16
    half = n // 2
    A = list(range(0, half))
    B = list(range(half, n))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
            
    return edges

# EVOLVE-BLOCK-END
