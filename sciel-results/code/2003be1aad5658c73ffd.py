# EVOLVE-BLOCK-START
def construct():
    """
    Returns edges of K_{7,8} on vertices 0..14.
    Part A = {0..6}, Part B = {7..14}.
    All A vertices have degree 8, all B vertices have degree 7.
    Degrees are disjoint between partitions.
    In K_{7,8}, any 3-edge path s-u-v-t has endpoints s,t in opposite partitions.
    Thus d(s) != d(t) always holds.
    Edge count = 56.
    """
    n = 15
    A = list(range(0, 7))
    B = list(range(7, 15))
    
    edges = []
    for u in A:
        for v in B:
            edges.append([u, v])
    
    # Ensure exact types: Python int for vertices, list of lists
    result = [[int(u), int(v)] for u, v in edges]
    return result
# EVOLVE-BLOCK-END
