"""Pentagon-free graphs without independent 9-sets (Ramsey number R(C5,K9))."""
# EVOLVE-BLOCK-START

def construct():
    """
    Construct a C5-free graph with no independent set of size 9, maximizing n.
    
    Returns the 8-disjoint-K4 construction with n=32, which is the best known
    valid graph. The 8 K4's are triangle-free (in fact, each is a clique K4),
    wait no, K4 is a clique, not triangle-free. But the union of disjoint K4's
    is C5-free because any cycle must be entirely within one component, and K4
    has no 5-cycle. The independence number is 8 (one vertex from each K4).
    """
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
