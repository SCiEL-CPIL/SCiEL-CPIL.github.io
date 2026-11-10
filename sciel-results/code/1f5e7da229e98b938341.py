import time

def construct():
    """
    Constructs a complete bipartite graph K_{7,8} on 15 vertices.
    Vertices 0-6 form part A, vertices 7-14 form part B.
    This graph has 56 edges.
    Validity check:
    - Any path of length 3 (s-u-v-t) in a bipartite graph has endpoints s and t in different parts.
    - In K_{7,8}, all vertices in A have degree 8, and all vertices in B have degree 7.
    - Since 8 != 7, deg(s) != deg(t) for all such paths.
    - Thus, the graph is valid.
    """
    n = 15
    edges = []
    # Part A: 0..6
    # Part B: 7..14
    for u in range(7):
        for v in range(7, 15):
            edges.append([u, v])
    return edges
