import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 52 decomposition
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2xp block
    # a_offset: (i, j) offset for A
    # b_offset: (j, k) offset for B
    # c_offset: (i, k) offset for C
    # p_block: number of columns in this block
    def add_block(a_i, a_j, b_j, b_k, c_i, c_k, p_block):
        nonlocal col
        # 1. Strassen for first 2 columns (k=0,1)
        # M1 = (a11 + a22)(b11 + b21)
        # M2 = (a21 + a22) b11
        # M3 = a11 (b12 - b22)
        # M4 = a22 (b21 - b11)
        # M5 = (a11 + a12) b22
        # M6 = (a21 - a11)(b11 + b12)
        # M7 = (a12 - a22)(b21 + b22)
        
        # Indices in local block:
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        
        # Term 1
        U[a_i*2 + a_j][col] = 1
        U[a_i*2 + a_j + 1][col] = 1
        V[b_j*2 + b_k][col] = 1
        V[b_j*2 + b_k + 1][col] = 1
        W[c_i*2 + c_k][col] = 1
        W[c_i*2 + c_k + 1][col] = 1
        W[(c_i+1)*2 + c_k][col] = 1
        W[(c_i+1)*2 + c_k + 1][col] = 1
        col += 1
        
        # Term 2
        U[a_i*2 + a_j + 1][col] = 1
        U[a_i*2 + a_j + 2][col] = 1
        V[b_j*2 + b_k][col] = 1
        W[c_i*2 + c_k + 1][col] = 1
        W[(c_i+1)*2 + c_k + 1][col] = 1
        col += 1
        
        # Term 3
        U[a_i*2 + a_j][col] = 1
        V[b_j*2 + b_k + 1][col] = 1
        V[b_j*2 + b_k + 2][col] = -1
        W[c_i*2 + c_k + 1][col] = 1
        W[(c_i+1)*2 + c_k + 1][col] = -1
        col += 1
        
        # Term 4
        U[a_i*2 + a_j + 1][col] = 1
        V[b_j*2 + b_k + 1][col] = -1
        V[b_j*2 + b_k][col] = 1
        W[c_i*2 + c_k][col] = -1
        W[(c_i+1)*2 + c_k][col] = 1
        col += 1
        
        # Term 5
        U[a_i*2 + a_j][col] = 1
        U[a_i*2 + a_j + 1][col] = 1
        V[b_j*2 + b_k + 1][col] = 1
        W[c_i*2 + c_k + 1][col] = 1
        W[(c_i+1)*2 + c_k + 1][col] = 1
        col += 1
        
        # Term 6
        U[a_i*2 + a_j + 1][col] = 1
        U[a_i*2 + a_j][col] = -1
        V[b_j*2 + b_k][col] = 1
        V[b_j*2 + b_k + 1][col] = 1
        W[c_i*2 + c_k][col] = 1
        W[(c_i+1)*2 + c_k][col] = 1
        col += 1
        
        # Term 7
        U[a_i*2 + a_j + 1][col] = 1
        U[a_i*2 + a_j][col] = -1
        V[b_j*2 + b_k + 1][col] = 1
        V[b_j*2 + b_k + 2][col] = 1
        W[c_i*2 + c_k + 1][col] = -1
        W[(c_i+1)*2 + c_k + 1][col] = 1
        col += 1
        
        # 2. Standard for remaining columns
        for kk in range(2, p_block):
            # N1 = a11 b1k
            U[a_i*2 + a_j][col] = 1
            V[b_j*2 + b_k + kk][col] = 1
            W[c_i*2 + c_k + kk][col] = 1
            col += 1
            
            # N2 = a12 b2k
            U[a_i*2 + a_j + 1][col] = 1
            V[b_j*2 + b_k + 1 + kk][col] = 1
            W[c_i*2 + c_k + kk][col] = 1
            col += 1
            
            # N3 = a21 b1k
            U[a_i*2 + a_j + 2][col] = 1
            V[b_j*2 + b_k + kk][col] = 1
            W[(c_i+1)*2 + c_k + kk][col] = 1
            col += 1
            
            # N4 = a22 b2k
            U[a_i*2 + a_j + 3][col] = 1
            V[b_j*2 + b_k + 1 + kk][col] = 1
            W[(c_i+1)*2 + c_k + kk][col] = 1
            col += 1

    # Block 1: A(0,0:2), B(0:2, 0:4) -> C(0:2, 0:4)
    # a_i=0, a_j=0, b_j=0, b_k=0, c_i=0, c_k=0, p_block=4
    add_block(0, 0, 0, 0, 0, 0, 4)
    
    # Block 2: A(0,2:4), B(2:4, 0:4) -> C(0:2, 0:4)
    # a_i=0, a_j=2, b_j=2, b_k=0, c_i=0, c_k=0, p_block=4
    add_block(0, 2, 2, 0, 0, 0, 4)
    
    # Block 3: A(0,0:2), B(0:2, 4:7) -> C(0:2, 4:7)
    # a_i=0, a_j=0, b_j=0, b_k=4, c_i=0, c_k=4, p_block=3
    add_block(0, 0, 0, 4, 0, 4, 3)
    
    # Block 4: A(0,2:4), B(2:4, 4:7) -> C(0:2, 4:7)
    # a_i=0, a_j=2, b_j=2, b_k=4, c_i=0, c_k=4, p_block=3
    add_block(0, 2, 2, 4, 0, 4, 3)
    
    return {"U": U, "V": V, "W": W}
