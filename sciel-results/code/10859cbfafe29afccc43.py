import time

def construct():
    """
    Constructs a dense odd-order graph on 15 vertices avoiding equal-degree path endpoints.
    Uses a direct construction of the complete bipartite graph K_{7,8}.

    Reasoning:
    In a complete bipartite graph K_{a,b}, vertices in partition A have degree b,
    and vertices in partition B have degree a.
    A path of length 3 s-u-v-t must alternate partitions: s in A, u in B, v in A, t in B
    OR s in B, u in A, v in B, t in A.
    In the first case, deg(s)=b and deg(t)=a.
    In the second case, deg(s)=a and deg(t)=b.
    If a != b, then deg(s) != deg(t) always holds.
    For n=15, we choose a=7, b=8.
    a=7, b=8 => 7 != 8.
    Number of edges = 7 * 8 = 56.
    """
    n = 15
    size_a = 7
    size_b = 8

    edges = []
    for u in range(size_a):
        for v in range(size_a, n):
            edges.append([u, v])

    return edges
