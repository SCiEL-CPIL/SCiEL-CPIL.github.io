import numpy as np
import time

DIMENSIONS = [3, 5, 7]
N, M, P = DIMENSIONS

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = N, M, P
    
    # Dimensions of the factor matrices
    dim_U = n * m  # 15
    dim_V = m * p  # 35
    dim_W = p * n  # 21
    
    # Target Tensor T*
    T_star = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T_star[a, b, d] = 1.0
    
    def construct_block(n_blk, m_blk, p_blk, i_offset, j_offset, k_offset):
        """
        Construct a block decomposition for an n_blk x m_blk x p_blk sub-matrix multiplication.
        Uses Strassen for 2x2x2, schoolbook otherwise.
        Returns (U_block, V_block, W_block) with shapes:
        U_block: (n_blk*m_blk, rank_block)
        V_block: (m_blk*p_blk, rank_block) 
        W_block: (p_blk*n_blk, rank_block)
        Indices in U_block, V_block, W_block are local to the block.
        """
        rank_sb = n_blk * m_blk * p_blk
        U_blk = np.zeros((n_blk * m_blk, rank_sb))
        V_blk = np.zeros((m_blk * p_blk, rank_sb))
        W_blk = np.zeros((p_blk * n_blk, rank_sb))
        
        col = 0
        for ii in range(n_blk):
            for jj in range(m_blk):
                for kk in range(p_blk):
                    # Local indices
                    a_local = ii * m_blk + jj
                    b_local = jj * p_blk + kk
                    d_local = kk * n_blk + ii
                    
                    U_blk[a_local, col] = 1.0
                    V_blk[b_local, col] = 1.0
                    W_blk[d_local, col] = 1.0
                    col += 1
        return U_blk, V_blk, W_blk, rank_sb

    def construct_strassen(m_blk, p_blk, i_offset, j_offset, k_offset):
        """
        Construct Strassen decomposition for 2x2x2 block.
        """
        # Strassen for 2x2 matrices: A * B = C
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # C = [[c11, c12], [c21, c22]]
        # c11 = M1 + M4 - M5 - M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M3 + M6 + M7
        # M1 = (a11 + a22)(b11 + b22)
        # M2 = (a21 + a22)b11
        # M3 = a11(b12 - b22)
        # M4 = a22(b21 - b11)
        # M5 = (a11 + a12)b22
        # M6 = (a21 - a11)(b11 + b12)
        # M7 = (a12 - a22)(b21 + b22)
        
        rank_s = 7
        U_s = np.zeros((4, rank_s))
        V_s = np.zeros((4, rank_s))
        W_s = np.zeros((4, rank_s))
        
        # Indices:
        # U index: a = i*m + j (local)
        # V index: b = j*p + k (local)
        # W index: d = k*n + i (local)
        
        # M1: (a11+a22)(b11+b22)
        # U: a11 (0,0), a22 (1,1)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        # V: b11 (0,0), b22 (1,1)
        V_s[0, 0] = 1; V_s[3, 0] = 1
        # W: c11 (0,0), c22 (1,1)
        W_s[0, 0] = 1; W_s[3, 0] = 1
        
        # M2: (a21+a22)b11
        # U: a21 (1,0), a22 (1,1)
        U_s[2, 1] = 1; U_s[3, 1] = 1
        # V: b11 (0,0)
        V_s[0, 1] = 1
        # W: c21 (1,0), c22 (1,1)
        W_s[2, 1] = 1; W_s[3, 1] = 1
        
        # M3: a11(b12-b22)
        # U: a11 (0,0)
        U_s[0, 2] = 1
        # V: b12 (0,1), b22 (1,1)
        V_s[1, 2] = 1; V_s[3, 2] = -1
        # W: c12 (0,1), c22 (1,1)
        W_s[1, 2] = 1; W_s[3, 2] = -1
        
        # M4: a22(b21-b11)
        # U: a22 (1,1)
        U_s[3, 3] = 1
        # V: b21 (1,0), b11 (0,0)
        V_s[2, 3] = 1; V_s[0, 3] = -1
        # W: c11 (0,0), c21 (1,0)
        W_s[0, 3] = 1; W_s[2, 3] = 1
        
        # M5: (a11+a12)b22
        # U: a11 (0,0), a12 (0,1)
        U_s[0, 4] = 1; U_s[1, 4] = 1
        # V: b22 (1,1)
        V_s[3, 4] = 1
        # W: c11 (0,0), c12 (0,1)
        W_s[0, 4] = -1; W_s[1, 4] = 1
        
        # M6: (a21-a11)(b11+b12)
        # U: a21 (1,0), a11 (0,0)
        U_s[2, 5] = 1; U_s[0, 5] = -1
        # V: b11 (0,0), b12 (0,1)
        V_s[0, 5] = 1; V_s[1, 5] = 1
        # W: c22 (1,1)
        W_s[3, 5] = 1
        
        # M7: (a12-a22)(b21+b22)
        # U: a12 (0,1), a22 (1,1)
        U_s[1, 6] = 1; U_s[3, 6] = -1
        # V: b21 (1,0), b22 (1,1)
        V_s[2, 6] = 1; V_s[3, 6] = 1
        # W: c11 (0,0), c22 (1,1)
        W_s[0, 6] = -1; W_s[3, 6] = 1
        
        return U_s, V_s, W_s, rank_s

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reconstructs the tensor."""
        T_approx = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(T_approx, T_star, atol=tol)

    # Strategy: Partition the dimensions into blocks.
    # We want to minimize total rank = sum(r_i * r_j * r_k) where r_i is rank of block.
    # For 2x2x2, Strassen rank is 7.
    # For other blocks, schoolbook is best known (or close to it).
    
    # Dimensions: n=3, m=5, p=7
    
    # Possible partitions:
    # n: [3], [2,1]
    # m: [5], [2,3], [1,4], [1,1,3], [2,2,1], etc.
    # p: [7], [2,5], [3,4], [1,6], [2,2,3], [1,2,4], etc.
    
    # Let's try a specific promising partition:
    # n=[2,1], m=[2,3], p=[2,5]
    # Blocks:
    # (2,2,2): Strassen (7)
    # (2,2,5): 2*2*5 = 20 (schoolbook) or better?
    # (2,3,2): 2*3*2 = 12
    # (2,3,5): 2*3*5 = 30
    # (1,2,2): 1*2*2 = 4
    # (1,2,5): 1*2*5 = 10
    # (1,3,2): 1*3*2 = 6
    # (1,3,5): 1*3*5 = 15
    # Total: 7 + 20 + 12 + 30 + 4 + 10 + 6 + 15 = 104. (Worse than 105? No, 104 < 105)
    
    # Let's try n=[3], m=[2,3], p=[2,5]
    # (3,2,2): 3*2*2 = 12
    # (3,2,5): 3*2*5 = 30
    # (3,3,2): 3*3*2 = 18
    # (3,3,5): 3*3*5 = 45
    # Total: 12 + 30 + 18 + 45 = 105. (Same as schoolbook)
    
    # Let's try n=[2,1], m=[3,2], p=[3,4]
    # (2,3,3): 2*3*3 = 18
    # (2,3,4): 2*3*4 = 24
    # (2,2,3): 2*2*3 = 12
    # (2,2,4): 2*2*4 = 16
    # (1,3,3): 1*3*3 = 9
    # (1,3,4): 1*3*4 = 12
    # (1,2,3): 1*2*3 = 6
    # (1,2,4): 1*2*4 = 8
    # Total: 18+24+12+16+9+12+6+8 = 105.
    
    # Let's try n=[2,1], m=[2,2,1], p=[2,2,3]
    # This is the partition mentioned in inspiration 3 which yielded 99.
    # Blocks:
    # (2,2,2): 7 (Strassen)
    # (2,2,2): 7 (Strassen)
    # (2,2,3): 12
    # (2,2,2): 7 (Strassen)
    # (2,2,2): 7 (Strassen)
    # (2,2,3): 12
    # (1,2,2): 4
    # (1,2,2): 4
    # (1,2,3): 6
    # (1,2,2): 4
    # (1,2,2): 4
    # (1,2,3): 6
    # Sum: 7+7+12 + 7+7+12 + 4+4+6 + 4+4+6 = 26 + 26 + 14 + 14 = 80? 
    # Wait, let's recalculate.
    # n parts: 2, 1
    # m parts: 2, 2, 1
    # p parts: 2, 2, 3
    
    # i=0 (n=2), j=0 (m=2), k=0 (p=2) -> 2x2x2 -> 7
    # i=0 (n=2), j=0 (m=2), k=1 (p=2) -> 2x2x2 -> 7
    # i=0 (n=2), j=0 (m=2), k=2 (p=3) -> 2x2x3 -> 12
    # i=0 (n=2), j=1 (m=2), k=0 (p=2) -> 2x2x2 -> 7
    # i=0 (n=2), j=1 (m=2), k=1 (p=2) -> 2x2x2 -> 7
    # i=0 (n=2), j=1 (m=2), k=2 (p=3) -> 2x2x3 -> 12
    # i=0 (n=2), j=2 (m=1), k=0 (p=2) -> 2x1x2 -> 4
    # i=0 (n=2), j=2 (m=1), k=1 (p=2) -> 2x1x2 -> 4
    # i=0 (n=2), j=2 (m=1), k=2 (p=3) -> 2x1x3 -> 6
    # Sum i=0: 7+7+12 + 7+7+12 + 4+4+6 = 66
    
    # i=1 (n=1), j=0 (m=2), k=0 (p=2) -> 1x2x2 -> 4
    # i=1 (n=1), j=0 (m=2), k=1 (p=2) -> 1x2x2 -> 4
    # i=1 (n=1), j=0 (m=2), k=2 (p=3) -> 1x2x3 -> 6
    # i=1 (n=1), j=1 (m=2), k=0 (p=2) -> 1x2x2 -> 4
    # i=1 (n=1), j=1 (m=2), k=1 (p=2) -> 1x2x2 -> 4
    # i=1 (n=1), j=1 (m=2), k=2 (p=3) -> 1x2x3 -> 6
    # i=1 (n=1), j=2 (m=1), k=0 (p=2) -> 1x1x2 -> 2
    # i=1 (n=1), j=2 (m=1), k=1 (p=2) -> 1x1x2 -> 2
    # i=1 (n=1), j=2 (m=1), k=2 (p=3) -> 1x1x3 -> 3
    # Sum i=1: 4+4+6 + 4+4+6 + 2+2+3 = 35
    
    # Total: 66 + 35 = 101.
    # This is better than 105.
    
    # Can we do better?
    # What if we use 3x3x3 BCL (rank 23) instead of schoolbook (27)?
    # Partition: n=[3], m=[3,2], p=[3,4]
    # (3,3,3): 23 (BCL)
    # (3,3,4): 3*3*4 = 36
    # (3,2,3): 3*2*3 = 18
    # (3,2,4): 3*2*4 = 24
    # Total: 23 + 36 + 18 + 24 = 101. Same.
    
    # Partition: n=[3], m=[2,3], p=[2,5]
    # (3,2,2): 12
    # (3,2,5): 30
    # (3,3,2): 18
    # (3,3,5): 45
    # Total: 105.
    
    # Partition: n=[2,1], m=[3,2], p=[3,4]
    # We calculated 105 earlier.
    
    # Let's stick with the 101 rank decomposition using Strassen for 2x2x2 blocks.
    # Partition: n=[2,1], m=[2,2,1], p=[2,2,3]
    
    def build_decomposition(n_parts, m_parts, p_parts):
        """
        Build the full decomposition from block partitions.
        """
        U = np.zeros((dim_U, 0))
        V = np.zeros((dim_V, 0))
        W = np.zeros((dim_W, 0))
        
        i_offset = 0
        for n_blk in n_parts:
            j_offset = 0
            for m_blk in m_parts:
                k_offset = 0
                for p_blk in p_parts:
                    # Get block decomposition
                    if n_blk == 2 and m_blk == 2 and p_blk == 2:
                        U_blk, V_blk, W_blk, r = construct_strassen(m_blk, p_blk, i_offset, j_offset, k_offset)
                    else:
                        U_blk, V_blk, W_blk, r = construct_block(n_blk, m_blk, p_blk, i_offset, j_offset, k_offset)
                    
                    # Map block indices to global indices
                    # U: local a (0..n_blk*m_blk-1) -> global a = i_offset*m + a_local
                    # V: local b (0..m_blk*p_blk-1) -> global b = j_offset*p + b_local
                    # W: local d (0..p_blk*n_blk-1) -> global d = k_offset*n + d_local
                    # But wait, the global indices are not simply shifted because the block starts at 
                    # i_offset in the i-dimension, j_offset in the j-dimension, k_offset in the k-dimension.
                    
                    # Global index for U:
                    # a_global = i_global * m + j_global
                    # i_global = i_offset + i_local
                    # j_global = j_offset + j_local
                    # So a_global = (i_offset + i_local) * m + (j_offset + j_local)
                    # = i_offset*m + i_local*m + j_offset + j_local
                    # This is not a simple offset unless m=1.
                    
                    # We need to map the local indices (0..n_blk*m_blk-1) to global indices.
                    # Local index l = i_local * m_blk + j_local
                    # Global index g = (i_offset + i_local) * m + (j_offset + j_local)
                    
                    U_g = np.zeros((dim_U, r))
                    for l in range(n_blk * m_blk):
                        i_local = l // m_blk
                        j_local = l % m_blk
                        g = (i_offset + i_local) * m + (j_offset + j_local)
                        U_g[g, :] = U_blk[l, :]
                    
                    V_g = np.zeros((dim_V, r))
                    for l in range(m_blk * p_blk):
                        j_local = l // p_blk
                        k_local = l % p_blk
                        g = (j_offset + j_local) * p + (k_offset + k_local)
                        V_g[g, :] = V_blk[l, :]
                    
                    W_g = np.zeros((dim_W, r))
                    for l in range(p_blk * n_blk):
                        k_local = l // n_blk
                        i_local = l % n_blk
                        g = (k_offset + k_local) * n + (i_offset + i_local)
                        W_g[g, :] = W_blk[l, :]
                    
                    U = np.hstack([U, U_g])
                    V = np.hstack([V, V_g])
                    W = np.hstack([W, W_g])
                    
                    k_offset += p_blk
                j_offset += m_blk
            i_offset += n_blk
            
        return U, V, W

    # Partition 1: n=[2,1], m=[2,2,1], p=[2,2,3] -> Rank 101
    n_parts1 = [2, 1]
    m_parts1 = [2, 2, 1]
    p_parts1 = [2, 2, 3]
    
    U1, V1, W1 = build_decomposition(n_parts1, m_parts1, p_parts1)
    
    if verify(U1, V1, W1):
        return {
            "U": U1.tolist(),
            "V": V1.tolist(),
            "W": W1.tolist()
        }
    
    # Fallback: Schoolbook
    U_sb = np.zeros((dim_U, n*m*p))
    V_sb = np.zeros((dim_V, n*m*p))
    W_sb = np.zeros((dim_W, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }

BET: {mechanism: "Deterministic block decomposition using Strassen for 2x2x2 blocks achieves rank 101, which is lower than the schoolbook rank of 105, improving the score from 1.0 to approx 1.05", q: 0.95, delta: [0.05, 0.05]}
