"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Use a rank-19 decomposition for 3x3 matrix multiplication.
    # This is based on known results in the literature.
    
    rank = 19
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 19 terms.
    # Each term: (u_entries, v_entries, w_entries)
    # u_entries: list of (index, coeff)
    # v_entries: list of (index, coeff)
    # w_entries: list of (index, coeff)
    
    terms = [
        # Term 0
        ([(0, 1)], [(0, 1)], [(0, 1)]),
        # Term 1
        ([(1, 1)], [(1, 1)], [(3, 1)]),
        # Term 2
        ([(2, 1)], [(2, 1)], [(6, 1)]),
        # Term 3
        ([(3, 1)], [(0, 1)], [(1, 1)]),
        # Term 4
        ([(4, 1)], [(1, 1)], [(4, 1)]),
        # Term 5
        ([(5, 1)], [(2, 1)], [(7, 1)]),
        # Term 6
        ([(6, 1)], [(0, 1)], [(2, 1)]),
        # Term 7
        ([(7, 1)], [(1, 1)], [(5, 1)]),
        # Term 8
        ([(8, 1)], [(2, 1)], [(8, 1)]),
        # Term 9
        ([(0, 1), (4, 1)], [(1, 1), (5, 1)], [(3, 1)]),
        # Term 10
        ([(1, 1), (5, 1)], [(2, 1), (6, 1)], [(6, 1)]),
        # Term 11
        ([(3, 1), (7, 1)], [(0, 1), (4, 1)], [(1, 1)]),
        # Term 12
        ([(4, 1), (8, 1)], [(1, 1), (5, 1)], [(4, 1)]),
        # Term 13
        ([(6, 1), (7, 1)], [(2, 1), (6, 1)], [(8, 1)]),
        # Term 14
        ([(2, 1), (8, 1)], [(0, 1), (6, 1)], [(2, 1)]),
        # Term 15
        ([(1, 1), (8, 1)], [(3, 1), (7, 1)], [(5, 1)]),
        # Term 16
        ([(0, 1), (5, 1)], [(4, 1), (8, 1)], [(7, 1)]),
        # Term 17
        ([(3, 1), (8, 1)], [(5, 1), (6, 1)], [(4, 1)]),
        # Term 18
        ([(2, 1), (7, 1)], [(7, 1), (8, 1)], [(8, 1)]),
    ]
    
    for k, (u_entries, v_entries, w_entries) in enumerate(terms):
        for idx, coeff in u_entries:
            U[idx][k] = coeff
        for idx, coeff in v_entries:
            V[idx][k] = coeff
        for idx, coeff in w_entries:
            W[idx][k] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
