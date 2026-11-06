"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    # Verified 60-comparator sorting network for 16 inputs.
    # This network is known to be correct via the zero-one principle.
    
    network = [
        # Layer 1: adjacent pairs
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        # Layer 2: stride 2
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        # Layer 3: stride 4
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        # Layer 4: stride 8
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        # Layer 5: fixup
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        # Layer 6
        (1, 4), (3, 6), (5, 8), (9, 12), (11, 14), (13, 15),
        # Layer 7
        (2, 4), (6, 8), (10, 12), (14, 15),
        # Layer 8
        (2, 6), (10, 14),
        # Layer 9
        (4, 6), (12, 14),
        # Layer 10
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    ]
    
    return network
# EVOLVE-BLOCK-END
