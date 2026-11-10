"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 16}

def construct():
    n = P['n']
    # Split vertices into two equal partitions for K_{n/2, n/2}
    # This maximizes edges in a triangle-free graph (Mantel's Theorem)
    # and satisfies the condition that every edge is essential for diameter 2
    # because removing any edge in a connected triangle-free graph with diam 2
    # creates a pair at distance >= 3.
    
    mid = n // 2
    partition_a = list(range(mid))
    partition_b = list(range(mid, n))
    
    edges = []
    for u in partition_a:
        for v in partition_b:
            edges.append([u, v])
            
    return edges
# EVOLVE-BLOCK-END
