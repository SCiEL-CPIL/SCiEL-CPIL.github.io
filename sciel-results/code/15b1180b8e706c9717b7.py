import time

def construct():
    """
    Returns a graph with n=32 vertices, no 5-cycle, and no independent set of size 9.
    Construction: 8 disjoint copies of K4.
    n = 8 * 4 = 32.
    Independence number = 8 (one from each K4).
    C5-free: K4 has no C5, and disjoint union has no cycles spanning components.
    
    This is the best known construction for R(C5, K9) > 32.
    Theoretical upper bound for n is 32 (since R(C5, K9) = 33).
    
    Note: The complement of this graph (complete 8-partite K_{4,4,...,4}) contains C5,
    so it is not a valid solution. The disjoint union of K4s is the optimal structure
    in this family.
    """
    n = 32
    edges = []
    
    # 8 blocks of 4 vertices (K4)
    for b in range(8):
        base = b * 4
        # Add all edges within the block (K4)
        # Vertices in block: base, base+1, base+2, base+3
        edges.append([base, base + 1])
        edges.append([base, base + 2])
        edges.append([base, base + 3])
        edges.append([base + 1, base + 2])
        edges.append([base + 1, base + 3])
        edges.append([base + 2, base + 3])
    
    result = {'n': n, 'edges': edges}
    
    return result
