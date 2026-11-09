import time

def construct():
    """
    Returns a simple graph on n vertices (1 <= n <= 35) that is C5-free 
    and has no independent set of size 9 (alpha <= 8).
    Maximizes n.
    
    Known valid construction: 8 disjoint K4 cliques.
    n = 32.
    alpha = 8.
    C5-free: Yes.
    """
    # 8 disjoint K4s
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
