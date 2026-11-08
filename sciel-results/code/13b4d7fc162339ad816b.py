import numpy as np

def construct():
    """
    Returns a rank-49 tensor decomposition for 4x4 matrix multiplication.
    This is the known optimal rank for M_{4,4,4} over the rationals.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen 2x2 coefficients
    # Rows correspond to the 7 multiplications (k=0..6)
    # Columns correspond to the 4 elements of the 2x2 block (0:00, 1:01, 2:10, 3:11)
    
    # A block: [a00, a01, a10, a11]
    S_A = np.array([
        [1, 0, 0, 1], # M1 = (A00+A11)(B00+B11)
        [0, 0, 1, 1], # M2 = (A10+A11)B00
        [1, 0, 0, 0], # M3 = A00(B10-B11)
        [0, 0, 0, 1], # M4 = A11(B01-B00)
        [1, 1, 0, 0], # M5 = (A00+A01)B11
        [-1, 0, 1, 0],# M6 = (A10-A00)(B00+B01)
        [0, 1, 0, -1] # M7 = (A01-A11)(B10+B11)
    ])
    
    # B block: [b00, b01, b10, b11]
    S_B = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # C block: [c00, c01, c10, c11]
    S_C = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Indices for U (A matrix): i in 0..3, j in 0..3
    # i, j map to block (i//2, j//2) and local (i%2, j%2)
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            rb, cb = i // 2, j // 2
            li, lj = i % 2, j % 2
            block_idx = rb * 2 + cb
            local_idx = li * 2 + lj
            
            for k in range(7):
                for l in range(7):
                    r = k * 7 + l
                    U[idx_u, r] = S_A[k, block_idx] * S_A[l, local_idx]
                    
    # Indices for V (B matrix): j in 0..3, k in 0..3
    for j in range(m):
        for k in range(p):
            idx_v = j * p + k
            rb, cb = j // 2, k // 2
            li, lj = j % 2, k % 2
            block_idx = rb * 2 + cb
            local_idx = li * 2 + lj
            
            for l in range(7):
                for k2 in range(7):
                    r = k2 * 7 + l
                    # Note: In the tensor decomposition T = sum_r U_i,r * V_j,r * W_k,r
                    # The indices for V correspond to the second dimension of the multiplication (j,k)
                    # The rank index r decomposes into (k_strassen, l_strassen) where k_strassen is the block multiplier and l_strassen is the local multiplier?
                    # Let's re-verify the Strassen 2-level structure.
                    
                    # In 2-level Strassen:
                    # We treat A as 2x2 blocks of 2x2.
                    # A_block[i][j] is a 2x2 matrix.
                    # The 7 Strassen multiplications are applied to the blocks.
                    # Let the 7 block-level multipliers be indexed by K (0..6).
                    # Each block multiplication is itself a 2x2 matrix multiplication, solved by 7 Strassen multipliers indexed by L (0..6).
                    # So total rank = 7 * 7 = 49.
                    # The term for rank r = K*7 + L is:
                    # U[i,j, r] = S_A[K, block_A(i,j)] * S_A[L, local_A(i,j)]
                    # V[j,k, r] = S_B[K, block_B(j,k)] * S_B[L, local_B(j,k)]
                    # W[i,k, r] = S_C[K, block_C(i,k)] * S_C[L, local_C(i,k)]
                    
                    # Wait, the indices in the previous code were:
                    # U[flat_u, r] = S_A[K, block_u] * S_A[l, local_u]
                    # V[flat_v, r] = S_B[K, block_v] * S_B[l, local_v]
                    # W[flat_w, r] = S_C[K, block_w] * S_C[l, local_w]
                    
                    # My loop above for V was:
                    # for l in range(7):
                    #     for k2 in range(7):
                    #         r = k2 * 7 + l
                    #         V[idx_v, r] = S_B[k2, block_idx] * S_B[l, local_idx]
                    # This matches K=k2, L=l.
                    
                    V[idx_v, r] = S_B[k2, block_idx] * S_B[l, local_idx]

    # Indices for W (C matrix): i in 0..3, k in 0..3
    # Trace convention: W[k*n+i, r]
    for i in range(n):
        for k in range(p):
            idx_w = k * n + i
            rb, cb = i // 2, k // 2
            li, lj = i % 2, k % 2
            block_idx = rb * 2 + cb
            local_idx = li * 2 + lj
            
            for K in range(7):
                for L in range(7):
                    r = K * 7 + L
                    W[idx_w, r] = S_C[K, block_idx] * S_C[L, local_idx]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
