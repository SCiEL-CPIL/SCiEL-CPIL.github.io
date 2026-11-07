import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    """
    Construct edge-disjoint Hamiltonian cycles on the 21-vertex host graph.
    Returns a list of cycles (each a list of vertices).
    """
    n = 21
    # The graph is a circulant graph C_21(1, 2, 5).
    # It has 63 edges total (21 vertices * 6 degree / 2).
    # A Hamiltonian cycle uses 21 edges.
    # Max number of edge-disjoint Hamiltonian cycles is 63 / 21 = 3.
    
    # We construct 3 cycles using steps 1, 2, and 5.
    # Step s cycle: 0, s, 2s, ..., 20s (mod 21)
    # Edges are of the form {i, i+s} and {i, i-s} (which is {i, i-(21-s)})
    # For s=1: edges {i, i+1} and {i, i-1} -> offsets 1, 20
    # For s=2: edges {i, i+2} and {i, i-2} -> offsets 2, 19
    # For s=5: edges {i, i+5} and {i, i-5} -> offsets 5, 16
    # These sets of offsets are disjoint: {1,20}, {2,19}, {5,16}.
    # Thus the cycles are edge-disjoint.
    
    def gen_cycle(step):
        return [(i * step) % n for i in range(n)]
    
    c1 = gen_cycle(1)
    c2 = gen_cycle(2)
    c3 = gen_cycle(5)
    
    return [c1, c2, c3]
# EVOLVE-BLOCK-END
