import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Known optimal or near-optimal graphs
    known_graphs = {}
    
    # Petersen Graph: k=3, g=5, n=10
    if k == 3 and g == 5:
        n = 10
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 6), (1, 7), (1, 8), (1, 9),
            (2, 6), (2, 7), (2, 8), (2, 9),
            (3, 6), (3, 7), (3, 8), (3, 9),
            (4, 6), (4, 7), (4, 8), (4, 9),
            (5, 6), (5, 7), (5, 8), (5, 9)
        ]
        # This is K_{5,5} minus a perfect matching? No, Petersen is K_{5,5} minus a 1-factor.
        # Let's use the standard Petersen edges.
        # Outer pentagon: 0-1-2-3-4-0
        # Inner pentagram: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        return n, edges

    # Heawood Graph: k=3, g=6, n=14
    if k == 3 and g == 6:
        n = 14
        # Incidence graph of Fano plane
        # Points 0-6, Lines 7-13
        # Line i contains points...
        # This is complex to hardcode without errors.
        # I will skip Heawood if I'm not sure.
        pass

    # Clebsch Graph: k=5, g=5, n=16
    if k == 5 and g == 5:
        n = 16
        # Clebsch graph is the folded 5-cube.
        # Vertices are binary strings of length 5 with even parity? No, 16 vertices.
        # It is the Kneser graph K(5,2) complement? No.
        # Clebsch graph is the graph of the 5-dimensional hypercube with antipodal pairs identified.
        # Vertices: 0..15
        # i ~ j if popcount(i xor j) == 1 or popcount(i xor j) == 4
        edges = []
        for i in range(16):
            for j in range(i + 1, 16):
                xor = i ^ j
                pc = bin(xor).count('1')
                if pc == 1 or pc == 4:
                    edges.append((i, j))
        return n, edges

    # Fall back to LUW construction
    # ... (seed code) ...
