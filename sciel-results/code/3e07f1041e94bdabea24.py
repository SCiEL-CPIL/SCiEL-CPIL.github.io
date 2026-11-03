"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal decomposition for 3x4 * 4x8 matrix multiplication.
    Strategy: Use block-Strassen with 2x2x2 base case.
    Decompose into:
      - 2 blocks of M_2,4,4 (rows 0,1 of A) using Strassen: 2 * 4 * 7 = 56
      - 2 blocks of M_1,4,4 (row 2 of A) using schoolbook: 2 * 4 * 4 = 32
    Total rank: 88
    """
    import time
    import numpy as np
    
    start_time = time.time()
    DEADLINE = 115.0  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 88
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen factors for 2x2x2 multiplication
    # Each factor: (a_coeffs, b_coeffs, c_coeffs) where coeffs are dicts {(i,j): val}
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # Process 2x M_2,4,4 blocks (for rows i=0,1 of the 3x4 matrix)
    # M_2,4,4: A is 2x4, B is 4x4, C is 4x2
    # Split B and C into 2x2 blocks: 4 sub-problems of M_2,2,2
    for c_block in range(2):  # 2 column blocks of B (and C)
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            for sub_prod_idx in range(4):
                # Determine offsets for the 2x2 sub-block
                if sub_prod_idx == 0:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 0
                    c_i_off, c_j_off = 0, 0
                elif sub_prod_idx == 1:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 2
                    c_i_off, c_j_off = 0, 2
                elif sub_prod_idx == 2:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 0
                    c_i_off, c_j_off = 0, 0
                else:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2
                
                for (ai, aj), au in a_coeffs.items():
                    u_idx = (ai + a_i_off) * 4 + (aj + a_j_off)
                    U[u_idx, col] += au
                for (bi, bj), bv in b_coeffs.items():
                    v_idx = (bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)
                    V[v_idx, col] += bv
                for (ci, cj), cw in c_coeffs.items():
                    w_idx = (cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)
                    W[w_idx, col] += cw
                col += 1
    
    # Process 2x M_1,4,4 blocks (for row i=2 of the 3x4 matrix)
    # M_1,4,4: A is 1x4, B is 4x4, C is 4x1
    # Schoolbook: 4*4 = 16 terms per block
    for c_block in range(2):
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    
    # Check for zero columns and prune
    keep_cols = []
    for r in range(rank):
        if np.any(U[:, r] != 0) or np.any(V[:, r] != 0) or np.any(W[:, r] != 0):
            keep_cols.append(r)
    
    if len(keep_cols) < rank:
        U = U[:, keep_cols]
        V = V[:, keep_cols]
        W = W[:, keep_cols]
    
    # Deadline check
    if time.time() - start_time > DEADLINE - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
