import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition U, V, W exactly decomposes the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    # Use a loop to avoid memory explosion if rank is large, though here it's small
    for r in range(U.shape[1]):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1
                
    return np.array_equal(T, T_exp)

def _schoolbook(n, m, p):
    """
    Returns U, V, W for schoolbook multiplication with rank n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen.
    U, V, W are 4x7 matrices.
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 1, 0, 0, 1, 0, -1], # A01
        [0, 0, 0, 0, 0, 1, 1],  # A10
        [1, 0, -1, 1, 0, 0, -1] # A11
    ], dtype=np.int64)
    
    V = np.array([
        [1, 0, 0, -1, 0, 1, 0], # B00
        [0, 1, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ], dtype=np.int64)
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ], dtype=np.int64)
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Construct rank 49 decomposition for 4x4x4 matrix multiplication using recursive Strassen.
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=np.int64)
    V = np.zeros((n*n, rank), dtype=np.int64)
    W = np.zeros((n*n, rank), dtype=np.int64)
    
    u2, v2, w2 = _strassen_2x2()
    
    # Strassen 7 multiplications for 2x2 blocks
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 -= M1
    # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7, C10 += M7
    
    # Definition of the 7 top-level multiplications
    # Each entry is (a_blocks, b_blocks, c_blocks)
    # a_blocks: list of (coeff, block_id) where block_id is 0,1,2,3 for 00,01,10,11
    # Same for b and c.
    
    mults = [
        # M1
        ([(1, 0), (1, 3)], [(1, 0), (1, 3)], [(1, 0), (-1, 3)]),
        # M2
        ([(1, 2), (1, 3)], [(1, 0)], [(1, 2), (-1, 3)]),
        # M3
        ([(1, 0)], [(1, 1), (-1, 3)], [(1, 1), (1, 3)]),
        # M4
        ([(1, 3)], [(1, 2), (-1, 0)], [(1, 0), (1, 2)]),
        # M5
        ([(1, 0), (1, 1)], [(1, 3)], [(-1, 0), (1, 1)]),
        # M6
        ([(1, 2), (-1, 0)], [(1, 0), (1, 1)], [(1, 3)]),
        # M7
        ([(1, 1), (-1, 3)], [(1, 2), (1, 3)], [(1, 0), (1, 2)])
    ]
    
    # Helper to map block id and local index to full index
    # Block id: 0 (00), 1 (01), 2 (10), 3 (11)
    # Local index: 0 (00), 1 (01), 2 (10), 3 (11)
    def map_index(block_id, local_idx, dim):
        # block_id determines the offset in the 4x4 matrix
        # 0: row 0-1, col 0-1
        # 1: row 0-1, col 2-3
        # 2: row 2-3, col 0-1
        # 3: row 2-3, col 2-3
        r_offset = (block_id // 2) * 2
        c_offset = (block_id % 2) * 2
        
        # local_idx determines the offset within the 2x2 block
        lr = local_idx // 2
        lc = local_idx % 2
        
        r = r_offset + lr
        c = c_offset + lc
        return r * dim + c

    # Fill the matrices
    for m_idx, (a_blocks, b_blocks, c_blocks) in enumerate(mults):
        # For each of the 7 inner Strassen multiplications
        for s_idx in range(7):
            col = m_idx * 7 + s_idx
            
            # Accumulate contributions to U, V, W
            # Note: We are effectively summing the outer products of the linear combinations.
            # U_full[row, col] = sum_{a_block, a_local} coeff_a * u2[a_local, s_idx] * (1 if a_local maps to row)
            # This is equivalent to:
            # For each a_block in a_blocks:
            #   For each a_local in 0..3:
            #     row = map_index(a_block, a_local, n)
            #     U[row, col] += coeff_a * u2[a_local, s_idx]
            
            for coeff_a, a_block in a_blocks:
                for a_local in range(4):
                    if u2[a_local, s_idx] != 0:
                        row = map_index(a_block, a_local, n)
                        U[row, col] += coeff_a * u2[a_local, s_idx]
                        
            for coeff_b, b_block in b_blocks:
                for b_local in range(4):
                    if v2[b_local, s_idx] != 0:
                        row = map_index(b_block, b_local, n) # V dimension is also 4x4 -> 16
                        V[row, col] += coeff_b * v2[b_local, s_idx]
                        
            for coeff_c, c_block in c_blocks:
                for c_local in range(4):
                    if w2[c_local, s_idx] != 0:
                        row = map_index(c_block, c_local, n) # W dimension is also 4x4 -> 16
                        W[row, col] += coeff_c * w2[c_local, s_idx]
                        
    return U, V, W

def construct():
    deadline = time.time() + 110.0 # Safety margin
    n, m, p = DIMENSIONS
    
    # Strategy:
    # 1. Decompose 4x4x4 part (k=0..3) using Strassen (Rank 49)
    # 2. Decompose 4x4x1 part (k=4) using Schoolbook (Rank 16)
    # Total Rank = 65
    
    U44, V44, W44 = _build_4x4x4_strassen()
    
    # Verify 4x4x4 part
    if not _verify(U44, V44, W44, n, m, 4):
        # Fallback to schoolbook for 4x4x4 if Strassen construction failed
        U44, V44, W44 = _schoolbook(n, m, 4)
        
    U1, V1, W1 = _schoolbook(n, m, 1)
    
    # Combine
    U = np.hstack([U44, U1])
    V = np.hstack([V44, V1])
    W = np.hstack([W44, W1])
    
    # Verify full 4x4x5
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # If verification fails (should not happen), fallback to full schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
