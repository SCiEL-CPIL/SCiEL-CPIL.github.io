import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank is 46
    rank = 46
    U = [[0] * rank for _ in range(n)]
    V = [[0] * rank for _ in range(m)]
    W = [[0] * rank for _ in range(p)]
    
    # Strassen coefficients for 2x2x2
    # U (2x7), V (2x7), W (2x7)
    # Indices 0,1 for local i,j,k
    S_U = [
        [1, 0, 1, 0, 1, -1, 1],
        [1, 1, 0, 1, 0, 1, -1]
    ]
    S_V = [
        [1, 1, 1, -1, 0, 1, 0],
        [1, 0, -1, 1, 1, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0]
    ]
    
    # Helper to add Strassen block
    def add_strassen(i_start, j_start, k_start, r_start):
        for r in range(7):
            for i_local in range(2):
                U[i_start + i_local][r_start + r] += S_U[i_local][r]
            for j_local in range(2):
                V[j_start + j_local][r_start + r] += S_V[j_local][r]
            for k_local in range(2):
                W[k_start + k_local][r_start + r] += S_W[k_local][r]
                
    # Helper to add rank-2 block for 1x2x6
    # i_start is the single row index
    # j_start is the start of the 2 cols in A (and 2 rows in B)
    # k_start is the start of the 6 cols in B (and 6 cols in C)
    def add_rank2(i_idx, j_start, k_start, r_start):
        # Term 1: j=0
        # U[i_idx] = 1
        # V[j_start] = 1
        # W[k_start + k] = 1 for all k in 0..5
        U[i_idx][r_start] = 1
        V[j_start][r_start] = 1
        for k in range(6):
            W[k_start + k][r_start] = 1
            
        # Term 2: j=1
        U[i_idx][r_start + 1] = 1
        V[j_start + 1][r_start + 1] = 1
        for k in range(6):
            W[k_start + k][r_start + 1] = 1

    # Block 1: j in {0, 1}
    # Sub-block 1.1: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen(0, 0, 0, 0)
    # Sub-block 1.2: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen(0, 0, 2, 7)
    # Sub-block 1.3: i in {0,1}, j in {0,1}, k in {4,5}
    add_strassen(0, 0, 4, 14)
    # Sub-block 1.4: i in {2}, j in {0,1}, k in {0..5}
    add_rank2(2, 0, 0, 21)
    
    # Block 2: j in {2, 3}
    # Sub-block 2.1: i in {0,1}, j in {2,3}, k in {0,1}
    add_strassen(0, 2, 0, 23)
    # Sub-block 2.2: i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen(0, 2, 2, 30)
    # Sub-block 2.3: i in {0,1}, j in {2,3}, k in {4,5}
    add_strassen(0, 2, 4, 37)
    # Sub-block 2.4: i in {2}, j in {2,3}, k in {0..5}
    add_rank2(2, 2, 0, 44)
    
    return {"U": U, "V": V, "W": W}
