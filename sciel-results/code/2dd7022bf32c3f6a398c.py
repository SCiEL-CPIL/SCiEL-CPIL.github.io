import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define Strassen 2x2 factors
    # U2 shape (4, 7), V2 shape (4, 7), W2 shape (4, 7)
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0 (M1)
    # u1: A00, A11 -> idx 0, 3
    U2[0, 0] = 1
    U2[3, 0] = 1
    # v1: B00, B11 -> idx 0, 3
    V2[0, 0] = 1
    V2[3, 0] = 1
    # w1: C00, C11 -> idx 0, 3
    W2[0, 0] = 1
    W2[3, 0] = 1
    
    # r=1 (M2)
    # u2: A10, A11 -> idx 2, 3
    U2[2, 1] = 1
    U2[3, 1] = 1
    # v2: B00 -> idx 0
    V2[0, 1] = 1
    # w2: C10 -> idx 1
    W2[1, 1] = 1
    
    # r=2 (M3)
    # u3: A00 -> idx 0
    U2[0, 2] = 1
    # v3: B01, -B11 -> idx 1, 3
    V2[1, 2] = 1
    V2[3, 2] = -1
    # w3: C01, -C11 -> idx 2, 3
    W2[2, 2] = 1
    W2[3, 2] = -1
    
    # r=3 (M4)
    # u4: A11 -> idx 3
    U2[3, 3] = 1
    # v4: B10, -B00 -> idx 2, 0
    V2[2, 3] = 1
    V2[0, 3] = -1
    # w4: C00, C10 -> idx 0, 1
    W2[0, 3] = 1
    W2[1, 3] = 1
    
    # r=4 (M5)
    # u5: A00, A01 -> idx 0, 1
    U2[0, 4] = 1
    U2[1, 4] = 1
    # v5: B11 -> idx 3
    V2[3, 4] = 1
    # w5: -C00, C01 -> idx 0, 2
    W2[0, 4] = -1
    W2[2, 4] = 1
    
    # r=5 (M6)
    # u6: A10, -A00 -> idx 2, 0
    U2[2, 5] = 1
    U2[0, 5] = -1
    # v6: B00, B01 -> idx 0, 1
    V2[0, 5] = 1
    V2[1, 5] = 1
    # w6: C11 -> idx 3
    W2[3, 5] = 1
    
    # r=6 (M7)
    # u7: A01, -A11 -> idx 1, 3
    U2[1, 6] = 1
    U2[3, 6] = -1
    # v7: B10, B11 -> idx 2, 3
    V2[2, 6] = 1
    V2[3, 6] = 1
    # w7: -C00, C11 -> idx 0, 3
    W2[0, 6] = -1
    W2[3, 6] = 1
    
    # Construct 4x4 factors via Kronecker product logic
    rank = 7 * 7
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for rb in range(7):
        for ri in range(7):
            r = rb * 7 + ri
            for I in range(4):
                for J in range(4):
                    # Decompose I, J into block and inner indices
                    ib = I // 2
                    ii = I % 2
                    jb = J // 2
                    ji = J % 2
                    
                    # U index: I*4 + J
                    u_idx = I * 4 + J
                    # U2 block index: ib*2 + jb
                    u2_block_idx = ib * 2 + jb
                    # U2 inner index: ii*2 + ji
                    u2_inner_idx = ii * 2 + ji
                    
                    U[u_idx, r] = U2[u2_block_idx, rb] * U2[u2_inner_idx, ri]
                    
            for J in range(4):
                for K in range(4):
                    jb = J // 2
                    ji = J % 2
                    kb = K // 2
                    ki = K % 2
                    
                    v_idx = J * 4 + K
                    v2_block_idx = jb * 2 + kb
                    v2_inner_idx = ji * 2 + ki
                    
                    V[v_idx, r] = V2[v2_block_idx, rb] * V2[v2_inner_idx, ri]
                    
            for K in range(4):
                for I in range(4):
                    kb = K // 2
                    ki = K % 2
                    ib = I // 2
                    ii = I % 2
                    
                    w_idx = K * 4 + I
                    w2_block_idx = kb * 2 + ib
                    w2_inner_idx = ki * 2 + ii
                    
                    W[w_idx, r] = W2[w2_block_idx, rb] * W2[w2_inner_idx, ri]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
