import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    rank = 22
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def apply_term(u_sparsity, v_sparsity, w_sparsity, col_idx):
        """
        Adds a rank-1 term to the decomposition.
        u_sparsity: list of (index, value) for U
        v_sparsity: list of (index, value) for V
        w_sparsity: list of (index, value) for W
        """
        for idx, val in u_sparsity:
            U[idx][col_idx] += val
        for idx, val in v_sparsity:
            V[idx][col_idx] += val
        for idx, val in w_sparsity:
            W[idx][col_idx] += val

    # The decomposition is composed of:
    # 1. Strassen multiplication for A[0:2, 0:2] * B[0:2, 0:2] (Rank 7)
    # 2. Strassen multiplication for A[0:2, 0:2] * B[0:2, 2:4] (Rank 7)
    # 3. Outer product for A[0:2, 2:3] * B[2:3, 0:4] (Rank 8)
    
    # Block 1: Strassen for cols 0-1 of A and rows 0-1 of B, output cols 0-1
    # U indices for A[0:2, 0:2]: 0, 1, 3, 4 (a11, a12, a21, a22)
    # V indices for B[0:2, 0:2]: 0, 1, 3, 4 (b11, b12, b21, b22)
    # W indices for C[0:2, 0:2]: 0, 1, 2, 3 (c11, c12, c21, c22)
    
    apply_term(
        [(0, 1), (4, 1)], 
        [(0, 1), (5, 1)], 
        [(0, 1), (3, 1)], 
        0
    )
    apply_term(
        [(3, 1), (4, 1)], 
        [(0, 1)], 
        [(1, 1), (3, -1)], 
        1
    )
    apply_term(
        [(0, 1)], 
        [(1, 1), (5, -1)], 
        [(2, 1), (3, 1)], 
        2
    )
    apply_term(
        [(4, 1)], 
        [(4, 1), (0, -1)], 
        [(0, 1), (1, 1)], 
        3
    )
    apply_term(
        [(0, 1), (1, 1)], 
        [(5, 1)], 
        [(0, -1), (2, 1)], 
        4
    )
    apply_term(
        [(3, 1), (0, -1)], 
        [(0, 1), (1, 1)], 
        [(3, 1)], 
        5
    )
    apply_term(
        [(1, 1), (4, -1)], 
        [(4, 1), (5, 1)], 
        [(0, 1)], 
        6
    )

    # Block 2: Strassen for cols 0-1 of A and rows 0-1 of B, output cols 2-3
    # V indices for B[0:2, 2:4]: 2, 3, 6, 7 (b13, b14, b23, b24)
    # W indices for C[0:2, 2:4]: 4, 5, 6, 7 (c13, c14, c23, c24)
    
    apply_term(
        [(0, 1), (4, 1)], 
        [(2, 1), (7, 1)], 
        [(4, 1), (7, 1)], 
        7
    )
    apply_term(
        [(3, 1), (4, 1)], 
        [(2, 1)], 
        [(5, 1), (7, -1)], 
        8
    )
    apply_term(
        [(0, 1)], 
        [(3, 1), (7, -1)], 
        [(6, 1), (7, 1)], 
        9
    )
    apply_term(
        [(4, 1)], 
        [(6, 1), (2, -1)], 
        [(4, 1), (5, 1)], 
        10
    )
    apply_term(
        [(0, 1), (1, 1)], 
        [(7, 1)], 
        [(4, -1), (6, 1)], 
        11
    )
    apply_term(
        [(3, 1), (0, -1)], 
        [(2, 1), (3, 1)], 
        [(7, 1)], 
        12
    )
    apply_term(
        [(1, 1), (4, -1)], 
        [(6, 1), (7, 1)], 
        [(4, 1)], 
        13
    )

    # Block 3: Outer product for col 2 of A and row 2 of B
    # U indices for A[0:2, 2:3]: 2, 5 (a13, a23)
    # V indices for B[2:3, 0:4]: 8, 9, 10, 11 (b31, b32, b33, b34)
    # W indices for C[0:2, 0:4]: 0..7
    
    # a13 * b31 -> c11
    apply_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
    # a13 * b32 -> c12
    apply_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
    # a13 * b33 -> c13
    apply_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
    # a13 * b34 -> c14
    apply_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
    
    # a23 * b31 -> c21
    apply_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
    # a23 * b32 -> c22
    apply_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
    # a23 * b33 -> c23
    apply_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
    # a23 * b34 -> c24
    apply_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

    return {"U": U, "V": V, "W": W}
