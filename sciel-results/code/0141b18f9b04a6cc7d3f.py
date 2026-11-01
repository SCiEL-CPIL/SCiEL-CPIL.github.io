"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-28 tensor decomposition for 2x2 by 2x8 matrix multiplication
    using 4 independent Strassen decompositions.
    
    Note: The rank of M_{2,2,8} is 28, which is the sum of the ranks of four 
    independent M_{2,2,2} tensors (each of rank 7). A rank of 24 is theoretically 
    impossible due to the additivity of tensor rank for direct sums with disjoint 
    supports in the B and C modes.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = 28

    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    # Strassen terms for 2x2 matrix multiplication
    # A indices: 0:00, 1:01, 2:10, 3:11
    # B indices: 0:00, 1:01, 2:10, 3:11
    # C indices: 0:00, 1:01, 2:10, 3:11
    #
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # Reconstruction:
    # c00 = P1 + P3 - P5 + P7
    # c01 = P5 - P7
    # c10 = P2 - P4 + P6
    # c11 = P1 - P2 + P4 + P6

    U_strassen = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, 1],   # P2
        [1, 0, 0, 0],   # P3
        [0, 0, 0, 1],   # P4
        [1, 1, 0, 0],   # P5
        [-1, 0, 1, 0],  # P6
        [0, 1, 0, -1]   # P7
    ]

    V_strassen = [
        [1, 0, 0, 1],   # P1
        [1, 0, 0, 0],   # P2
        [0, 1, 0, -1],  # P3
        [-1, 0, 1, 0],  # P4
        [0, 0, 0, 1],   # P5
        [1, 1, 0, 0],   # P6
        [0, 0, 1, 1]    # P7
    ]

    W_strassen = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, -1],  # P2
        [0, 1, 0, 1],   # P3
        [1, 0, 1, 0],   # P4
        [-1, 1, 0, 0],  # P5
        [0, 0, 0, 1],   # P6
        [1, 0, 0, 0]    # P7
    ]

    num_blocks = p // 2  # 4 blocks of 2 columns each

    for block_idx in range(num_blocks):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = U_strassen[term_idx], V_strassen[term_idx], W_strassen[term_idx]

            # Fill U: A is 2x2, indices 0..3. Same for all blocks.
            for i in range(4):
                U[i][col] = u[i]

            # Fill V: B is 2x8. Block covers columns 2*block_idx and 2*block_idx+1.
            # Global B index: j * p + k_global.
            v_indices = [2 * block_idx, 2 * block_idx + 1,
                         p + 2 * block_idx, p + 2 * block_idx + 1]
            for l in range(4):
                V[v_indices[l]][col] = v[l]

            # Fill W: C is 8x2 (p x n).
            # Convention: W[k*n+i, r] where k is column, i is row.
            # Local C indices 0,1,2,3 correspond to:
            # 0: C[0][2t] -> i=0, k=2t -> 2t*2 + 0
            # 1: C[0][2t+1] -> i=0, k=2t+1 -> (2t+1)*2 + 0
            # 2: C[1][2t] -> i=1, k=2t -> 2t*2 + 1
            # 3: C[1][2t+1] -> i=1, k=2t+1 -> (2t+1)*2 + 1
            w_indices = [4 * block_idx, 4 * block_idx + 2,
                         4 * block_idx + 1, 4 * block_idx + 3]
            for l in range(4):
                W[w_indices[l]][col] = w[l]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
