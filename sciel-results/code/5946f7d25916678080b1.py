import numpy as np

DIMENSIONS = [3, 4, 5]

def _add_block(U, V, W, i_start, i_len, j_start, j_len, k_start, k_len, block_U, block_V, block_W, offset):
    """
    Adds a block decomposition to the global matrices.
    i_start, i_len: indices in U (row dimension n*m)
    j_start, j_len: indices in V (row dimension m*p)
    k_start, k_len: indices in W (row dimension p*n)
    
    block_U, block_V, block_W: local factor matrices.
    offset: starting column in global U, V, W.
    """
    n, m, p = DIMENSIONS
    
    # U local shape: (i_len * j_len, block_rank)
    # Global U row index: i * m + j
    # i is global row index of A (0..n-1)
    # j is global col index of A (0..m-1)
    
    # V local shape: (j_len * k_len, block_rank)
    # Global V row index: j * p + k
    # j is global col index of A (0..m-1)
    # k is global row index of B (0..p-1)
    
    # W local shape: (k_len * i_len, block_rank)
    # Global W row index: k * n + i
    # k is global row index of B (0..p-1)
    # i is global row index of A (0..n-1)
    
    for local_i in range(i_len):
        for local_j in range(j_len):
            global_i = i_start + local_i
            global_j = j_start + local_j
            u_idx = global_i * m + global_j
            for r in range(block_U.shape[1]):
                U[u_idx, offset + r] += block_U[local_i * j_len + local_j, r]
                
    for local_j in range(j_len):
        for local_k in range(k_len):
            global_j = j_start + local_j
            global_k = k_start + local_k
            v_idx = global_j * p + global_k
            for r in range(block_V.shape[1]):
                V[v_idx, offset + r] += block_V[local_j * k_len + local_k, r]
                
    for local_k in range(k_len):
        for local_i in range(i_len):
            global_k = k_start + local_k
            global_i = i_start + local_i
            w_idx = global_k * n + global_i
            for r in range(block_W.shape[1]):
                W[w_idx, offset + r] += block_W[local_k * i_len + local_i, r]

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_222():
    # Returns U, V, W for 2x2x2 with rank 7
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # m3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # m4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1; W[0, 3] = 1
    
    # m5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # m6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[3, 5] = 1
    
    # m7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[2, 6] = 1; W[1, 6] = 1
    
    # Correct W indices for 2x2x2:
    # W[k*n+i] -> W[k*2+i]
    # m1 contributes to c11 (0,0) and c22 (1,1)?
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Let's re-verify W contributions.
    # W[r, col] adds to C[i, k] where W index is k*2+i.
    # m1: c11 (0,0) and c22 (1,1).
    # W[0, 0] = 1 (c11)
    # W[3, 0] = 1 (c22) -> k=1, i=1 -> idx 3. Correct.
    
    # m2: c21 (1,0).
    # W[2, 1] = 1 (c21) -> k=1, i=0 -> idx 2. Correct.
    # Wait, m2 is (a21+a22)b11.
    # a21 is A[1,0], a22 is A[1,1].
    # b11 is B[0,0].
    # Result is C[1,0].
    # So W[2, 1] = 1.
    # My code above: W[2, 1] = 1; W[3, 1] = 1.
    # c22 also gets m2? No, c22 = m1 - m2 + ...
    # So W[3, 1] should be -1.
    
    # Let's rebuild W carefully.
    W = np.zeros((4, 7))
    # m1
    W[0, 0] = 1 # c11
    W[3, 0] = 1 # c22
    # m2
    W[2, 1] = 1 # c21
    W[3, 1] = -1 # c22
    # m3
    W[1, 2] = 1 # c12
    W[3, 2] = 1 # c22
    # m4
    W[2, 3] = 1 # c21
    W[0, 3] = 1 # c11
    # m5
    W[1, 4] = 1 # c12
    W[3, 4] = -1 # c22
    # m6
    W[0, 5] = 1 # c11
    W[3, 5] = 1 # c22
    # m7
    W[0, 6] = 1 # c11
    W[1, 6] = 1 # c12
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Blocks:
    # 1: i=0:2, j=0:2, k=0:3 (2x2x3)
    # 2: i=0:2, j=2:4, k=0:3 (2x2x3)
    # 3: i=2:3, j=0:2, k=0:3 (1x2x3)
    # 4: i=2:3, j=2:4, k=0:3 (1x2x3)
    # 5: i=0:2, j=0:2, k=3:5 (2x2x2)
    # 6: i=0:2, j=2:4, k=3:5 (2x2x2)
    # 7: i=2:3, j=0:2, k=3:5 (1x2x2)
    # 8: i=2:3, j=2:4, k=3:5 (1x2x2)
    
    blocks = [
        (0, 2, 0, 2, 0, 3), # 2x2x3
        (0, 2, 2, 2, 0, 3), # 2x2x3
        (2, 1, 0, 2, 0, 3), # 1x2x3
        (2, 1, 2, 2, 0, 3), # 1x2x3
        (0, 2, 0, 2, 3, 2), # 2x2x2
        (0, 2, 2, 2, 3, 2), # 2x2x2
        (2, 1, 0, 2, 3, 2), # 1x2x2
        (2, 1, 2, 2, 3, 2), # 1x2x2
    ]
    
    # Calculate total rank
    total_rank = 0
    block_decomps = []
    for b in blocks:
        i_s, i_l, j_s, j_l, k_s, k_l = b
        if i_l == 2 and j_l == 2 and k_l == 2:
            U_b, V_b, W_b = _strassen_222()
        else:
            U_b, V_b, W_b = _schoolbook(i_l, j_l, k_l)
        block_decomps.append((U_b, V_b, W_b))
        total_rank += U_b.shape[1]
        
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    offset = 0
    for idx, b in enumerate(blocks):
        i_s, i_l, j_s, j_l, k_s, k_l = b
        U_b, V_b, W_b = block_decomps[idx]
        _add_block(U, V, W, i_s, i_l, j_s, j_l, k_s, k_l, U_b, V_b, W_b, offset)
        offset += U_b.shape[1]
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
