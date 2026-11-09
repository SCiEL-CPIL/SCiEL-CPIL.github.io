import time

def construct():
    """
    Returns a C5-free graph with no independent set of size 9.
    
    The construction is 8 disjoint copies of K4, giving n=32.
    - Each K4 is a clique of 4 vertices.
    - The graph is C5-free (each component is K4, which has no 5-cycle).
    - The independence number is 8 (at most one vertex from each K4 can be in an independent set).
    - Thus alpha = 8 < 9.
    
    This certifies R(C5, K9) > 32.
    """
    n = 32
    edges = []
    for b in range(8):
        offset = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([offset + i, offset + j])
    return {'n': n, 'edges': edges}
