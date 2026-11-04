import numpy as np

DIMENSIONS = [5, 5, 5]

def _add_block(U, V, W, u_block, v_block, w_block, offset):
    """
    Add a block decomposition to the global matrices.
    U, V, W are global matrices.
    u_block, v_block, w_block are local decompositions.
    offset is the starting column index in the global matrices.
    """
    n_u, r_u = u_block.shape
    n_v, r_v = v_block.shape
    n_w, r_w = w_block.shape
    assert r_u == r_v == r_w
    rank = r_u
    
    # Global matrices are passed as lists of lists or numpy arrays.
    # We will assume they are numpy arrays for ease of manipulation.
    U[:, offset:offset+rank] += u_block
    V[:, offset:offset+rank] += v_block
    W[:, offset:offset+rank] += w_block

def _index_4x4(i, j):
    return i * 4 + j

def _index_5x5(i, j):
    return i * 5 + j

def _create_4x4_schoolbook():
    n = 4
    rank = n * n * n
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i*n+j, col] = 1
                V[j*n+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _create_4x1x4_outer():
    # A (4x1) * B (1x4) -> C (4x4)
    # C_ij = A_i B_j
    # Rank 1
    U = np.zeros((4, 1))
    V = np.zeros((4, 1))
    W = np.zeros((16, 1))
    U[:, 0] = 1
    V[:, 0] = 1
    W[:, 0] = 1
    return U, V, W

def _create_2x2x1_matvec():
    # A (2x2) * B (2x1) -> C (2x1)
    # C_i = sum_k A_ik B_k
    # Rank 4
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    col = 0
    for i in range(2):
        for k in range(2):
            U[i*2+k, col] = 1
            V[k, col] = 1
            W[i, col] = 1
            col += 1
    return U, V, W

def _create_4x4x1_matvec():
    # A (4x4) * B (4x1) -> C (4x1)
    # Split into two 2x2 blocks
    U, V, W = _create_2x2x1_matvec()
    U_full = np.zeros((16, 8))
    V_full = np.zeros((4, 8))
    W_full = np.zeros((4, 8))
    
    # Block 1: Top 2x2 of A, Top 2x1 of B -> Top 2x1 of C
    # A indices: 0,1,2,3 (rows 0-1, cols 0-1)
    # B indices: 0,1 (rows 0-1, col 0)
    # C indices: 0,1 (rows 0-1, col 0)
    
    # Map local indices to global 4x4 indices
    # Local A (2x2) row i, col k -> Global A row i, col k
    # Local B (2x1) row k -> Global B row k
    # Local C (2x1) row i -> Global C row i
    
    U_full[:4, :4] = U
    V_full[:2, :4] = V
    W_full[:2, :4] = W
    
    # Block 2: Bottom 2x2 of A, Bottom 2x1 of B -> Bottom 2x1 of C
    # Global A rows 2-3, cols 0-1 -> indices 4,5,6,7
    # Global B rows 2-3 -> indices 2,3
    # Global C rows 2-3 -> indices 2,3
    
    U_full[4:8, 4:8] = U
    V_full[2:4, 4:8] = V
    W_full[2:4, 4:8] = W
    
    return U_full, V_full, W_full

def _create_1x4x4_vecmat():
    # A (1x4) * B (4x4) -> C (1x4)
    # Symmetric to matvec
    U, V, W = _create_4x4x1_matvec()
    # Transpose logic?
    # A is 1x4. B is 4x4. C is 1x4.
    # C_j = sum_i A_i B_ij
    # This is different.
    
    U = np.zeros((4, 8))
    V = np.zeros((16, 8))
    W = np.zeros((4, 8))
    col = 0
    for j in range(4):
        for i in range(4):
            U[i, col] = 1
            V[i*4+j, col] = 1
            W[j, col] = 1
            col += 1
    # Rank 16?
    # C_j = sum_i A_i B_ij.
    # This is 4 dot products of length 4.
    # Rank 4*4=16.
    # My previous estimate of 8 was wrong.
    # For 4x4 * 4x1, I split into 2x2.
    # 2x2 * 2x1 has rank 4.
    # Two of them -> rank 8.
    # Why is 1x4 * 4x4 rank 16?
    # Because B is 4x4, not 4x1.
    # A (1x4) * B (4x4).
    # C is 1x4.
    # C_j = sum_i A_i B_ij.
    # For each j, we have a dot product of A and column j of B.
    # 4 independent dot products of length 4.
    # Rank 4*4 = 16.
    
    return U, V, W

def _create_1x4x1_dot():
    # A (1x4) * B (4x1) -> C (1x1)
    # Rank 4
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((1, 4))
    for i in range(4):
        U[i, i] = 1
        V[i, i] = 1
        W[0, i] = 1
    return U, V, W

def _create_1x1x1_scalar():
    U = np.array([[1]])
    V = np.array([[1]])
    W = np.array([[1]])
    return U, V, W

def _create_4x1x1_scale():
    # A (4x1) * B (1x1) -> C (4x1)
    # C_i = A_i B_0
    # Rank 4
    U = np.zeros((4, 4))
    V = np.zeros((1, 4))
    W = np.zeros((4, 4))
    for i in range(4):
        U[i, i] = 1
        V[0, i] = 1
        W[i, i] = 1
    return U, V, W

def _create_1x1x4_scale():
    # A (1x1) * B (1x4) -> C (1x4)
    # C_j = A_0 B_j
    # Rank 4
    U = np.zeros((1, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    for j in range(4):
        U[0, j] = 1
        V[j, j] = 1
        W[j, j] = 1
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    total_rank = 0
    # Calculate ranks
    rank_444 = 64
    rank_414 = 1
    rank_441 = 8
    rank_411 = 4
    rank_144 = 16
    rank_114 = 4
    rank_141 = 4
    rank_111 = 1
    
    # C11: 444 + 414
    r11 = rank_444 + rank_414
    # C12: 441 + 411
    r12 = rank_441 + rank_411
    # C21: 144 + 114
    r21 = rank_144 + rank_114
    # C22: 141 + 111
    r22 = rank_141 + rank_111
    
    total_rank = r11 + r12 + r21 + r22
    
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    offset = 0
    
    # C11
    U4, V4, W4 = _create_4x4_schoolbook()
    # Map 4x4 indices to 5x5
    # A11: rows 0-3, cols 0-3
    # Indices: 0,1,2,3, 5,6,7,8, 10,11,12,13, 15,16,17,18
    idx_A = [i*5+j for i in range(4) for j in range(4)]
    # B11: rows 0-3, cols 0-3
    idx_B = [i*5+j for i in range(4) for j in range(4)]
    # C11: rows 0-3, cols 0-3 -> W index k*5+i
    # W is 16x64. W[k*4+i, col]
    # Global W index: k*5+i
    # We need to place W4[k*4+i, col] to Global W[k*5+i, offset+col]
    
    U[idx_A, offset:offset+rank_444] = U4
    V[idx_B, offset:offset+rank_444] = V4
    
    W_block = np.zeros((25, rank_444))
    for i in range(4):
        for k in range(4):
            W_block[k*5+i, :] = W4[k*4+i, :]
    W[:, offset:offset+rank_444] += W_block
    offset += rank_444
    
    U4o, V4o, W4o = _create_4x1x4_outer()
    # A12: rows 0-3, col 4. Indices 4, 9, 14, 19
    idx_A12 = [i*5+4 for i in range(4)]
    # B21: row 4, cols 0-3. Indices 20, 21, 22, 23
    idx_B21 = [4*5+j for j in range(4)]
    # C11: rows 0-3, cols 0-3
    # W4o is 16x1. W4o[k*4+i]
    U[idx_A12, offset:offset+rank_414] = U4o
    V[idx_B21, offset:offset+rank_414] = V4o
    W_block = np.zeros((25, rank_414))
    for i in range(4):
        for k in range(4):
            W_block[k*5+i, :] = W4o[k*4+i, :]
    W[:, offset:offset+rank_414] += W_block
    offset += rank_414
    
    # C12
    U441, V441, W441 = _create_4x4x1_matvec()
    # A11: idx_A
    # B12: rows 0-3, col 4. Indices 4, 9, 14, 19
    idx_B12 = [i*5+4 for i in range(4)]
    # C12: rows 0-3, col 4. W index 4*5+i = 20+i
    U[idx_A, offset:offset+rank_441] = U441
    V[idx_B12, offset:offset+rank_441] = V441
    W_block = np.zeros((25, rank_441))
    # W441 is 4x8. W441[i, col]
    for i in range(4):
        W_block[20+i, :] = W441[i, :]
    W[:, offset:offset+rank_441] += W_block
    offset += rank_441
    
    U411, V411, W411 = _create_4x1x1_scale()
    # A12: idx_A12
    # B22: index 24
    U[idx_A12, offset:offset+rank_411] = U411
    V[[24], offset:offset+rank_411] = V411
    W_block = np.zeros((25, rank_411))
    # W411 is 4x4. W411[i, col]
    for i in range(4):
        W_block[20+i, :] = W411[i, :]
    W[:, offset:offset+rank_411] += W_block
    offset += rank_411
    
    # C21
    U144, V144, W144 = _create_1x4x4_vecmat()
    # A21: row 4, cols 0-3. Indices 20, 21, 22, 23
    idx_A21 = [4*5+j for j in range(4)]
    # B11: idx_B
    # C21: row 4, cols 0-3. W index k*5+4
    U[idx_A21, offset:offset+rank_144] = U144
    V[idx_B, offset:offset+rank_144] = V144
    W_block = np.zeros((25, rank_144))
    # W144 is 4x16. W144[j, col]
    for j in range(4):
        for k in range(4):
            # W144 corresponds to C_j.
            # My _create_1x4x4_vecmat returns W of shape (4, 16) where W[j, col] is for C_j.
            # We need to map this to W_block[k*5+4, col]
            W_block[k*5+4, :] = W144[j, :]
    W[:, offset:offset+rank_144] += W_block
    offset += rank_144
    
    U114, V114, W114 = _create_1x1x4_scale()
    # A22: index 24
    # B21: idx_B21
    U[[24], offset:offset+rank_114] = U114
    V[idx_B21, offset:offset+rank_114] = V114
    W_block = np.zeros((25, rank_114))
    # W114 is 4x4. W114[j, col]
    for j in range(4):
        W_block[j*5+4, :] = W114[j, :]
    W[:, offset:offset+rank_114] += W_block
    offset += rank_114
    
    # C22
    U141, V141, W141 = _create_1x4x1_dot()
    # A21: idx_A21
    # B12: idx_B12
    U[idx_A21, offset:offset+rank_141] = U141
    V[idx_B12, offset:offset+rank_141] = V141
    W_block = np.zeros((25, rank_141))
    # W141 is 1x4.
    W_block[24, :] = W141[0, :]
    W[:, offset:offset+rank_141] += W_block
    offset += rank_141
    
    U111, V111, W111 = _create_1x1x1_scalar()
    U[[24], offset:offset+rank_111] = U111
    V[[24], offset:offset+rank_111] = V111
    W_block = np.zeros((25, rank_111))
    W_block[24, :] = W111[0, :]
    W[:, offset:offset+rank_111] += W_block
    offset += rank_111
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
