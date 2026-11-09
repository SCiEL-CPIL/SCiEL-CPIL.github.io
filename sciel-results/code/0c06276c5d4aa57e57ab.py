"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # Hub 12 (even) connected to all other odd vertices except those already adjacent in path (11, 13)
    # This ensures bipartite structure (triangle-free) and diameter <= 4
    added_edges = [
        [12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
        [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]
    ]
    return added_edges
# EVOLVE-BLOCK-END
