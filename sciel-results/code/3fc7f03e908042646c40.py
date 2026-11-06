"""Pentagon-free graph with independence number < 9 for R(C5,K9) lower bound."""
# EVOLVE-BLOCK-START

def construct():
    # Construction: 8 disjoint cliques of size 4.
    # Each K4 is C5-free (no cycle of length 5 in a 4-vertex clique).
    # Independence number of each K4 is 1, so total alpha = 8 < 9.
    # Total vertices: 8 * 4 = 32.
    
    n = 32
    clique_size = 4
    num_cliques = n // clique_size  # 8

    edges = []
    for c in range(num_cliques):
        start = c * clique_size
        for i in range(start, start + clique_size):
            for j in range(i + 1, start + clique_size):
                edges.append([i, j])

    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
