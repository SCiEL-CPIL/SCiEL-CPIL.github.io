import time

def construct():
    # The task is to construct a simple undirected graph on 15 vertices (0..14)
    # maximizing edges such that no simple path of length 3 (s-u-v-t)
    # has endpoints s and t with equal degrees in the full graph.
    
    # A complete bipartite graph K_{m, n} with m + n = 15 and m != n
    # satisfies this condition.
    # In K_{m, n}, any path of length 3 has endpoints in opposite partitions.
    # Let A be partition of size m, B be partition of size n.
    # Vertices in A have degree n. Vertices in B have degree m.
    # Since m != n, the degrees of endpoints s (in A) and t (in B) are n and m,
    # which are distinct.
    
    # To maximize edges m*n subject to m+n=15 and m!=n,
    # we choose m=7, n=8 (or vice versa).
    # Edges = 7 * 8 = 56.
    
    n_vertices = 15
    size_a = 7
    size_b = 8
    
    # Partition A: 0 .. 6
    # Partition B: 7 .. 14
    
    edges = []
    
    for u in range(size_a):
        for v in range(size_a, n_vertices):
            edges.append([int(u), int(v)])
            
    return edges
