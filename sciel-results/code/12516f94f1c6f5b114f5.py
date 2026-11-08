"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 2x4 matrix by a 4x7 matrix.
    
    Uses a block decomposition based on Strassen's algorithm for 2x2x2 blocks.
    Rank 50.
    """
    n, m, p = DIMENSIONS
    rank = 50
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b00, b01, b10, b11]
        # c_idx: [c00, c01, c10, c11]
        
        # M1
        U[a_idx[0], col] += 1; U[a_idx[2], col] += 1
        V[b_idx[0], col] += 1; V[b_idx[2], col] += 1
        W[c_idx[0], col] += 1; W[c_idx[1], col] += 1
        W[c_idx[2], col] += 1; W[c_idx[3], col] += -1
        col += 1
        
        # M2
        U[a_idx[2], col] += 1; U[a_idx[3], col] += 1
        V[b_idx[0], col] += 1
        W[c_idx[2], col] += 1; W[c_idx[3], col] += -1
        col += 1
        
        # M3
        U[a_idx[0], col] += 1
        V[b_idx[1], col] += 1; V[b_idx[3], col] += -1
        W[c_idx[1], col] += 1; W[c_idx[3], col] += 1
        col += 1
        
        # M4
        U[a_idx[3], col] += 1
        V[b_idx[2], col] += 1; V[b_idx[0], col] += -1
        W[c_idx[0], col] += 1; W[c_idx[2], col] += 1
        col += 1
        
        # M5
        U[a_idx[0], col] += 1; U[a_idx[1], col] += 1
        V[b_idx[3], col] += 1
        W[c_idx[0], col] += -1; W[c_idx[1], col] += 1
        col += 1
        
        # M6
        U[a_idx[2], col] += 1; U[a_idx[0], col] += -1
        V[b_idx[0], col] += 1; V[b_idx[1], col] += 1
        W[c_idx[3], col] += 1
        col += 1
        
        # M7
        U[a_idx[1], col] += 1; U[a_idx[3], col] += -1
        V[b_idx[2], col] += 1; V[b_idx[3], col] += 1
        W[c_idx[0], col] += 1
        col += 1

    # Helper to add 2x2x1 block (rank 4)
    def add_2x2x1(a_idx, b_idx, c_idx):
        nonlocal col
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b0, b1]
        # c_idx: [c0, c1]
        
        # T1: a00 * b0 -> c0
        U[a_idx[0], col] += 1
        V[b_idx[0], col] += 1
        W[c_idx[0], col] += 1
        col += 1
        
        # T2: a01 * b1 -> c0
        U[a_idx[1], col] += 1
        V[b_idx[1], col] += 1
        W[c_idx[0], col] += 1
        col += 1
        
        # T3: a10 * b0 -> c1
        U[a_idx[2], col] += 1
        V[b_idx[0], col] += 1
        W[c_idx[1], col] += 1
        col += 1
        
        # T4: a11 * b1 -> c1
        U[a_idx[3], col] += 1
        V[b_idx[1], col] += 1
        W[c_idx[1], col] += 1
        col += 1

    # Block 1: A1 * B1
    a1_idx = [0, 1, 4, 5]
    
    b1_1_idx = [0, 1, 7, 8]
    c1_1_idx = [0, 1, 2, 3]
    add_strassen(a1_idx, b1_1_idx, c1_1_idx)
    
    b1_2_idx = [2, 3, 9, 10]
    c1_2_idx = [4, 5, 6, 7]
    add_strassen(a1_idx, b1_2_idx, c1_2_idx)
    
    b1_3_idx = [4, 5, 11, 12]
    c1_3_idx = [8, 9, 10, 11]
    add_strassen(a1_idx, b1_3_idx, c1_3_idx)
    
    b1_4_idx = [6, 13]
    c1_4_idx = [12, 13]
    add_2x2x1(a1_idx, b1_4_idx, c1_4_idx)
    
    # Block 2: A2 * B2
    a2_idx = [2, 3, 6, 7]
    
    b2_1_idx = [14, 15, 21, 22]
    c2_1_idx = [0, 1, 2, 3]
    add_strassen(a2_idx, b2_1_idx, c2_1_idx)
    
    b2_2_idx = [16, 17, 23, 24]
    c2_2_idx = [4, 5, 6, 7]
    add_strassen(a2_idx, b2_2_idx, c2_2_idx)
    
    b2_3_idx = [18, 19, 25, 26]
    c2_3_idx = [8, 9, 10, 11]
    add_strassen(a2_idx, b2_3_idx, c2_3_idx)
    
    b2_4_idx = [20, 27]
    c2_4_idx = [12, 13]
    add_2x2x1(a2_idx, b2_4_idx, c2_4_idx)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
