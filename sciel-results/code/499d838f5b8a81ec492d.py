import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # We split the p dimension into two blocks of size 3
    # Each block is a 2x3x3 multiplication
    # Rank of 2x3x3 schoolbook is 2*3*3 = 9
    # Total rank = 9 + 9 = 18
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook for 2x3x3
    # n=2, m=3, p_block=3
    n_block = 2
    m_block = 3
    p_block = 3
    
    def fill_block(offset_u, offset_v, offset_w):
        col = 0
        for i in range(n_block):
            for j in range(m_block):
                for k in range(p_block):
                    u_idx = i * m_block + j
                    v_idx = j * p_block + k
                    w_idx = k * n_block + i
                    U[u_idx][offset_u + col] = 1
                    V[v_idx][offset_v + col] = 1
                    W[w_idx][offset_w + col] = 1
                    col += 1

    # Block 1: k=0,1,2
    # V indices: j*p + k. For block 1, k is 0,1,2.
    # W indices: k*n + i. For block 1, k is 0,1,2.
    fill_block(0, 0, 0)
    
    # Block 2: k=3,4,5
    # V indices: j*p + k. For block 2, k is 3,4,5.
    # But our schoolbook V for 2x3x3 assumes k=0,1,2.
    # So we need to map the local k to global k.
    # Local k=0 -> Global k=3.
    # Local k=1 -> Global k=4.
    # Local k=2 -> Global k=5.
    
    # We can just modify the fill function to take a k_offset
    def fill_block_offset(offset_u, offset_v, offset_w, k_offset):
        col = 0
        for i in range(n_block):
            for j in range(m_block):
                for k_local in range(p_block):
                    k_global = k_local + k_offset
                    u_idx = i * m_block + j
                    v_idx = j * p + k_global
                    w_idx = k_global * n_block + i
                    U[u_idx][offset_u + col] = 1
                    V[v_idx][offset_v + col] = 1
                    W[w_idx][offset_w + col] = 1
                    col += 1

    # Reset and refill
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    fill_block_offset(0, 0, 0, 0)
    fill_block_offset(9, 9, 9, 3)
    
    return {"U": U, "V": V, "W": W}
