"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns tensor decomposition for matrix multiplication 2x3 * 3x9.
    Uses rank 30 decomposition.
    """
    n, m, p = 2, 3, 9
    
    U_rows = n * m  # 6
    V_rows = m * p  # 27
    W_rows = p * n  # 18
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_col = [0] * U_rows
        for idx, val in u_dict.items():
            u_col[idx] = val
        v_col = [0] * V_rows
        for idx, val in v_dict.items():
            v_col[idx] = val
        w_col = [0] * W_rows
        for idx, val in w_dict.items():
            w_col[idx] = val
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    # Part 1: A[0:2, 0:2] * B[0:2, 0:9]
    # A indices: i in {0,1}, j in {0,1} -> U[i*m + j]
    # B indices: j in {0,1}, k in {0..8} -> V[j*p + k]
    # C indices: i in {0,1}, k in {0..8} -> W[k*n + i]
    
    # Blocks of k: (0,1), (2,3), (4,5), (6,7), (8)
    
    # Strassen terms for 2x2 x 2x2
    # Local indices: j' in {0,1}, k' in {0,1}
    # Global j = j', Global k = k0 + k'
    
    def strassen_block(k0):
        # Term 1
        add_term(
            {0: 1, 4: 1},  # A00, A11
            {k0: 1, 9 + k0 + 1: 1},  # B0k0, B1k1
            {k0 * 2 + 0: 1, (k0 + 1) * 2 + 1: 1}  # C0k0, C1k1
        )
        # Term 2
        add_term(
            {3: 1, 4: 1},  # A10, A11
            {k0: 1},  # B0k0
            {k0 * 2 + 1: 1, (k0 + 1) * 2 + 1: -1}  # C1k0, C1k1
        )
        # Term 3
        add_term(
            {0: 1},  # A00
            {k0 + 1: 1, 9 + k0 + 1: -1},  # B0k1, B1k1
            {(k0 + 1) * 2 + 0: 1, (k0 + 1) * 2 + 1: 1}  # C0k1, C1k1
        )
        # Term 4
        add_term(
            {4: 1},  # A11
            {9 + k0: 1, k0: -1},  # B1k0, B0k0
            {k0 * 2 + 0: 1, k0 * 2 + 1: 1}  # C0k0, C1k0
        )
        # Term 5
        add_term(
            {0: 1, 1: 1},  # A00, A01
            {9 + k0 + 1: 1},  # B1k1
            {k0 * 2 + 0: -1, (k0 + 1) * 2 + 0: 1}  # C0k0, C0k1
        )
        # Term 6
        add_term(
            {3: 1, 0: -1},  # A10, A00
            {k0: 1, k0 + 1: 1},  # B0k0, B0k1
            {(k0 + 1) * 2 + 1: 1}  # C1k1
        )
        # Term 7
        add_term(
            {1: 1, 4: -1},  # A01, A11
            {9 + k0: 1, 9 + k0 + 1: 1},  # B1k0, B1k1
            {k0 * 2 + 0: 1}  # C0k0
        )
        
    for k0 in [0, 2, 4, 6]:
        strassen_block(k0)
        
    # Last block k=8 (2x1)
    # Rank 1
    add_term(
        {0: 1, 1: 1, 3: 1, 4: 1},  # A00, A01, A10, A11
        {8: 1, 17: 1},  # B08, B18 (9+8=17)
        {16: 1, 17: 1}  # C08 (8*2+0), C18 (8*2+1)
    )
    
    # Part 2: A[0:2, 2] * B[2, 0:9]
    # A indices: i in {0,1}, j=2 -> U[i*m + 2]
    # B indices: j=2, k in {0..8} -> V[2*p + k]
    # C indices: i in {0,1}, k in {0..8} -> W[k*n + i]
    
    # Rank 1
    u_dict = {2: 1, 5: 1}  # A02, A12
    v_dict = {18 + k: 1 for k in range(9)}  # B20..B28
    w_dict = {k * 2 + i: 1 for k in range(9) for i in range(2)}  # C00..C18
    
    add_term(u_dict, v_dict, w_dict)
    
    rank = len(U_cols)
    
    U = [[U_cols[r][i] for r in range(rank)] for i in range(U_rows)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(V_rows)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(W_rows)]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
