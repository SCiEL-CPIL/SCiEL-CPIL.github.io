"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    # The graph is a circulant graph C_21(1, 2, 5).
    # It decomposes into 3 edge-disjoint Hamiltonian cycles.
    # Cycle 1: step 1
    # Cycle 2: step 2
    # Cycle 3: step 5
    
    cycle1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    
    cycle2 = []
    cur = 0
    for _ in range(21):
        cycle2.append(cur)
        cur = (cur + 2) % 21
    
    cycle3 = []
    cur = 0
    for _ in range(21):
        cycle3.append(cur)
        cur = (cur + 5) % 21
        
    return [cycle1, cycle2, cycle3]
# EVOLVE-BLOCK-END
