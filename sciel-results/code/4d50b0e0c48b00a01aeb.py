import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    deadline = time.time() + 130
    
    # Winograd 2x2x2 decomposition (Rank 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.float64)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [1, 0, -1, 0, 0, 0, 1]
    ], dtype=np.float64)
    
    W2 = np.array([
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 1, 1, 0, 1],
        [0, 1, 0, 1, 0, -1, 1],
        [1, -1, 0, 0, 0, 1, 0]
    ], dtype=np.float64)
    
    # 2x4x2 rank-11 decomposition
    # A: 2x4 = [a0,a1,a2,a3; b0,b1,b2,b3]
    # B: 4x2 = [c0,d0; c1,d1; c2,d2; c3,d3]
    # C: 2x2 = [p,q;r,s]
    # 
    # Using structured decomposition with 11 products:
    U242 = np.array([
        [1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0],  # a0
        [1, -1,  1, -1,  0,  0,  0,  0,  0,  0,  0],  # a1
        [1,  1, -1, -1,  0,  0,  0,  0,  0,  0,  0],  # a2
        [1, -1, -1,  1,  0,  0,  0,  0,  0,  0,  0],  # a3
        [0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  0],  # b0
        [0,  0,  0,  0,  1, -1,  1, -1,  0,  0,  0],  # b1
        [0,  0,  0,  0,  1,  1, -1, -1,  0,  0,  0],  # b2
        [0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0],  # b3
    ], dtype=np.float64)
    
    V242 = np.array([
        [1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0],  # c0
        [1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0],  # c1
        [1,  1, -1, -1,  1,  1, -1, -1,  0,  0,  0],  # c2
        [1, -1, -1,  1,  1, -1, -1,  1,  0,  0,  0],  # c3
        [0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1],  # d0
        [0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  1],  # d1
        [0,  0,  0,  0,  0,  0,  0,  0,  1,  1, -1],  # d2
        [0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1],  # d3
    ], dtype=np.float64)
    
    W242 = np.array([
        [1,  1,  1,  1,  0,  0,  0,  0,  1,  1,  1],  # p (C00)
        [0,  0,  0,  0,  0,  0,  0,  0,  1, -1,  1],  # q (C01)
        [0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  0],  # r (C10)
        [0,  0,  0,  0,  1, -1,  1, -1,  1, -1, -1],  # s (C11)
    ], dtype=np.float64)
    
    # Verify 2x4x2 decomposition
    def verify_242():
        # C = A * B
        # C[0,0] = sum_j A[0,j]*B[j,0]
        # C[0,1] = sum_j A[0,j]*B[j,1]
        # C[1,0] = sum_j A[1,j]*B[j,0]
        # C[1,1] = sum_j A[1,j]*B[j,1]
        # Tensor T[i,j,k] where i in 0..1, j in 0..3, k in 0..1
        # T[i,j,k] = 1 if this is the (i,j,k) entry of the multiplication tensor
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (it's the identity tensor for matmul)
        # Reconstruction: sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] should equal 1 for all i,j,k? No.
        # The tensor is T[i,j,k] = delta (it's 1 for all valid i,j,k in the multiplication tensor)
        # Actually the multiplication tensor M[i,j,k] = 1 for all i,j,k
        # And M = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # So M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
        # Let's verify:
        T = np.ones((2, 4, 2))
        T_recon = np.einsum('ac,bc,dc->abd', U242, V242, W242)
        return np.allclose(T_recon, T, atol=1e-10)
    
    if not verify_242():
        # Fallback: use two 2x2x2 Winograd blocks
        # This means we'll use the original approach
        use_242 = False
    else:
        use_242 = True
    
    if use_242:
        # Build full 4x5x6 decomposition
        # j=0: 4x1x6 schoolbook (rank 24)
        # j=1..4: for each i_block (2 blocks of 2) and k_block (3 blocks of 2):
        #   2x4x2 block (rank 11)
        # Total: 24 + 2*3*11 = 24 + 66 = 90
        
        total_rank = 24 + 66
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col = 0
        
        # Part 1: j=0 (4x1x6), Rank 24
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
        
        # Part 2: j=1..4, using 2x4x2 blocks
        i_blocks = [(0, 1), (2, 3)]
        k_blocks = [(0, 1), (2, 3), (4, 5)]
        
        for i_start, i_end in i_blocks:
            for k_start, k_end in k_blocks:
                # 2x4x2 block: i in [i_start, i_start+1], j in [1,4], k in [k_start, k_start+1]
                # A indices (2x4): rows i_start, i_start+1; cols 1,2,3,4
                a_rows = [i_start, i_start + 1]
                a_cols = [1, 2, 3, 4]
                
                # B indices (4x2): rows 1,2,3,4; cols k_start, k_start+1
                b_rows = [1, 2, 3, 4]
                b_cols = [k_start, k_start + 1]
                
                # C indices (2x2): rows i_start, i_start+1; cols k_start, k_start+1
                c_rows = [i_start, i_start + 1]
                c_cols = [k_start, k_start + 1]
                
                for r in range(11):
                    # U: 8 entries (2x4)
                    for li in range(8):
                        ii = li // 4
                        jj = li % 4
                        u_idx = a_rows[ii] * m + a_cols[jj]
                        U[u_idx, col + r] += U242[li, r]
                    
                    # V: 8 entries (4x2)
                    for li in range(8):
                        jj = li // 2
                        kk = li % 2
                        v_idx = b_rows[jj] * p + b_cols[kk]
                        V[v_idx, col + r] += V242[li, r]
                    
                    # W: 4 entries (2x2)
                    for li in range(4):
                        kk = li // 2
                        ii = li % 2
                        w_idx = c_cols[kk] * n + c_rows[ii]
                        W[w_idx, col + r] += W242[li, r]
                
                col += 11
        
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        
        # Verify
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
        if not np.allclose(T_recon, T, atol=1e-10):
            # Fallback to schoolbook
            U, V, W = _schoolbook(n, m, p)
    else:
        # Use original approach with 2x2x2 Winograd blocks
        total_rank = 108
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        col = 0
        
        # Part 1: j=0 (4x1x6), Rank 24
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 0
                v_idx = 0 * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
        
        # Part 2: j=1..4 (4x4x6), Rank 84
        i_blocks = [(0, 1), (2, 3)]
        k_blocks = [(0, 1), (2, 3), (4, 5)]
        
        for i_start, i_end in i_blocks:
            for k_start, k_end in k_blocks:
                for j_offset in [0, 2]:
                    a_idx = [
                        i_start * m + (1 + j_offset),
                        i_start * m + (2 + j_offset),
                        (i_start + 1) * m + (1 + j_offset),
                        (i_start + 1) * m + (2 + j_offset)
                    ]
                    
                    b_idx = [
                        (1 + j_offset) * p + k_start,
                        (1 + j_offset) * p + (k_start + 1),
                        (2 + j_offset) * p + k_start,
                        (2 + j_offset) * p + (k_start + 1)
                    ]
                    
                    c_idx = [
                        k_start * n + i_start,
                        k_start * n + (i_start + 1),
                        (k_start + 1) * n + i_start,
                        (k_start + 1) * n + (i_start + 1)
                    ]
                    
                    for r in range(7):
                        for li in range(4):
                            U[a_idx[li], col + r] += U2[li, r]
                            V[b_idx[li], col + r] += V2[li, r]
                            W[c_idx[li], col + r] += W2[li, r]
                    col += 7
        
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        
        # Verify
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
        if not np.allclose(T_recon, T, atol=1e-10):
            U, V, W = _schoolbook(n, m, p)
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}


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
