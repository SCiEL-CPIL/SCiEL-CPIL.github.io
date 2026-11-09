"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a graph with 4 disjoint copies of K4 and 2 disjoint copies of the complement of C9.
    n = 34.
    Each K4 has alpha = 1. Total alpha from K4s = 4.
    Each complement of C9 has alpha = 2. Total alpha from comp(C9)s = 4.
    Total alpha = 8.
    K4 contains no 5-cycle.
    Complement of C9 contains no 5-cycle.
    Disjoint union contains no 5-cycle.
    """
    n = 34
    edges = []
    
    # 4 copies of K4
    # Vertices 0..15
    for b in range(4):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
        
    # 2 copies of complement of C9
    # Vertices 16..24 (first C9)
    # Vertices 25..33 (second C9)
    for b in range(2):
        base = 16 + 9 * b
        for i in range(9):
            for j in range(i + 1, 9):
                # Edge exists if i and j are NOT adjacent in C9
                # Adjacent in C9 if (j - i) % 9 == 1 or (i - j) % 9 == 1
                if (j - i) % 9 != 1 and (i - j) % 9 != 1:
                    edges.append([base + i, base + j])
                    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
