# EVOLVE-BLOCK-START
def construct():
    edges = []
    # Bipartite graph with parts U={0, ..., 9} and V={10, ..., 19}
    # Complete bipartite graph K_{10,10} has 100 edges.
    # Coloring: c(u, v) = (u + 10 * (v - 10)) % 20
    # This ensures that for any u1 != u2 and v1 != v2, the four edge colors
    # c(u1, v1), c(u1, v2), c(u2, v1), c(u2, v2) are distinct.
    # Proof:
    # c(u1, v1) = c(u2, v2) => u1 - u2 = 10(v2 - v1).
    # Since |u1 - u2| < 10 and |10(v2 - v1)| >= 10 (if v1 != v2), this is impossible.
    # c(u1, v2) = c(u2, v1) => u1 - u2 = 10(v1 - v2).
    # Similarly impossible.
    # Other equalities imply u1=u2 or v1=v2.
    
    for u in range(10):
        for v in range(10, 20):
            v_prime = v - 10
            c = (u + 10 * v_prime) % 20
            edges.append([u, v, c])
            
    return edges
# EVOLVE-BLOCK-END
