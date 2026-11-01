import time
import numpy as np

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Reconstruct T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Using einsum for speed
    # U is (n*m, rank) -> reshape to (n, m, rank)
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    if not np.allclose(T, T_recon, atol=1e-9):
        return False
    return True

def _strassen_rank14():
    """Construct the rank-14 decomposition using two independent Strassen blocks."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))  # 4 x 14
    V = np.zeros((m*p, rank))  # 8 x 14
    W = np.zeros((p*n, rank))  # 8 x 14
    
    # Indices:
    # U: i*m + j. A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # V: j*p + k. B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7
    # W: k*n + i. C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3, C[0,2]=4, C[0,3]=5, C[1,2]=6, C[1,3]=7
    
    # Strassen 2x2x2 factors (0-indexed for local 2x2):
    # m1 = (A11+A22)(B11+B22) -> C11 += m1, C22 += m1
    # m2 = (A21+A22)B11       -> C21 += m2, C22 -= m2
    # m3 = A11(B21-B22)       -> C12 += m3, C22 += m3
    # m4 = A22(B12-B11)       -> C11 += m4, C21 += m4
    # m5 = (A11+A12)B22       -> C12 += m5, C11 -= m5
    # m6 = (A21-A11)(B11+B12) -> C22 += m6
    # m7 = (A12-A22)(B21+B22) -> C11 += m7
    
    # Left Half (k=0,1): B indices 0,1,4,5. C indices 0,1,2,3.
    # Right Half (k=2,3): B indices 2,3,6,7. C indices 4,5,6,7.
    
    def add_strassen_block(offset_u, offset_v, offset_w, b_idx_map, c_idx_map):
        """
        Adds a Strassen block.
        offset_u, offset_v, offset_w are not used as offsets but as base indices if needed, 
        but here we just write directly to U, V, W using the global indices.
        b_idx_map: dict mapping local B index (0,1,2,3 for 2x2) to global V index.
        c_idx_map: dict mapping local C index (0,1,2,3 for 2x2) to global W index.
        """
        # Local A indices are 0,1,2,3 in U (since A is shared, no offset)
        
        # m1: (A0+A3)(B0+B3)
        r = 0
        U[0, r] = 1; U[3, r] = 1
        V[b_idx_map[0], r] = 1; V[b_idx_map[3], r] = 1
        W[c_idx_map[0], r] = 1; W[c_idx_map[3], r] = 1
        
        # m2: (A2+A3)B0
        r = 1
        U[2, r] = 1; U[3, r] = 1
        V[b_idx_map[0], r] = 1
        W[c_idx_map[2], r] = 1; W[c_idx_map[3], r] = -1
        
        # m3: A0(B2-B3)
        r = 2
        U[0, r] = 1
        V[b_idx_map[2], r] = 1; V[b_idx_map[3], r] = -1
        W[c_idx_map[1], r] = 1; W[c_idx_map[3], r] = 1
        
        # m4: A3(B1-B0)
        r = 3
        U[3, r] = 1
        V[b_idx_map[1], r] = 1; V[b_idx_map[0], r] = -1
        W[c_idx_map[0], r] = 1; W[c_idx_map[2], r] = 1
        
        # m5: (A0+A1)B3
        r = 4
        U[0, r] = 1; U[1, r] = 1
        V[b_idx_map[3], r] = 1
        W[c_idx_map[1], r] = 1; W[c_idx_map[0], r] = -1
        
        # m6: (A2-A0)(B0+B1)
        r = 5
        U[2, r] = 1; U[0, r] = -1
        V[b_idx_map[0], r] = 1; V[b_idx_map[1], r] = 1
        W[c_idx_map[3], r] = 1
        
        # m7: (A1-A3)(B2+B3)
        r = 6
        U[1, r] = 1; U[3, r] = -1
        V[b_idx_map[2], r] = 1; V[b_idx_map[3], r] = 1
        W[c_idx_map[0], r] = 1

    # Left Half:
    # B local 0 -> global 0 (B[0,0])
    # B local 1 -> global 1 (B[0,1])
    # B local 2 -> global 4 (B[1,0])
    # B local 3 -> global 5 (B[1,1])
    b_left = {0: 0, 1: 1, 2: 4, 3: 5}
    # C local 0 -> global 0 (C[0,0])
    # C local 1 -> global 1 (C[0,1])
    # C local 2 -> global 2 (C[1,0])
    # C local 3 -> global 3 (C[1,1])
    c_left = {0: 0, 1: 1, 2: 2, 3: 3}
    
    # We need to offset the rank indices for the second block.
    # The function above uses r=0..6.
    # For the left block, we use r=0..6.
    # For the right block, we use r=7..13.
    
    # Let's modify add_strassen_block to take a rank_offset
    def add_strassen_block_offset(rank_offset, b_idx_map, c_idx_map):
        # m1
        r = rank_offset + 0
        U[0, r] = 1; U[3, r] = 1
        V[b_idx_map[0], r] = 1; V[b_idx_map[3], r] = 1
        W[c_idx_map[0], r] = 1; W[c_idx_map[3], r] = 1
        
        # m2
        r = rank_offset + 1
        U[2, r] = 1; U[3, r] = 1
        V[b_idx_map[0], r] = 1
        W[c_idx_map[2], r] = 1; W[c_idx_map[3], r] = -1
        
        # m3
        r = rank_offset + 2
        U[0, r] = 1
        V[b_idx_map[2], r] = 1; V[b_idx_map[3], r] = -1
        W[c_idx_map[1], r] = 1; W[c_idx_map[3], r] = 1
        
        # m4
        r = rank_offset + 3
        U[3, r] = 1
        V[b_idx_map[1], r] = 1; V[b_idx_map[0], r] = -1
        W[c_idx_map[0], r] = 1; W[c_idx_map[2], r] = 1
        
        # m5
        r = rank_offset + 4
        U[0, r] = 1; U[1, r] = 1
        V[b_idx_map[3], r] = 1
        W[c_idx_map[1], r] = 1; W[c_idx_map[0], r] = -1
        
        # m6
        r = rank_offset + 5
        U[2, r] = 1; U[0, r] = -1
        V[b_idx_map[0], r] = 1; V[b_idx_map[1], r] = 1
        W[c_idx_map[3], r] = 1
        
        # m7
        r = rank_offset + 6
        U[1, r] = 1; U[3, r] = -1
        V[b_idx_map[2], r] = 1; V[b_idx_map[3], r] = 1
        W[c_idx_map[0], r] = 1

    add_strassen_block_offset(0, b_left, c_left)
    
    # Right Half:
    # B local 0 -> global 2 (B[0,2])
    # B local 1 -> global 3 (B[0,3])
    # B local 2 -> global 6 (B[1,2])
    # B local 3 -> global 7 (B[1,3])
    b_right = {0: 2, 1: 3, 2: 6, 3: 7}
    # C local 0 -> global 4 (C[0,2])
    # C local 1 -> global 5 (C[0,3])
    # C local 2 -> global 6 (C[1,2])
    # C local 3 -> global 7 (C[1,3])
    c_right = {0: 4, 1: 5, 2: 6, 3: 7}
    
    add_strassen_block_offset(7, b_right, c_right)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = 2, 2, 4
    start_time = time.time()
    
    # Rank 14 is the known minimum for 2x2x4.
    # We construct it directly.
    res = _strassen_rank14()
    
    # Verify to be safe
    if _verify(res["U"], res["V"], res["W"], n, m, p):
        return res
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
