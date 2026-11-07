"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

P = {'n': 16}

def construct():
    n = P['n']
    half = n // 2
    
    # The optimal graph is K_{8,8}, the unique extremal triangle-free graph on 16 vertices.
    # Any graph with more edges contains a triangle, making its edges non-critical.
    A = list(range(half))
    B = list(range(half, n))
    
    edges = [[u, v] for u in A for v in B]
    
    return edges
# EVOLVE-BLOCK-END
