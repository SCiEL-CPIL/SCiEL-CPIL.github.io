"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # 2x2x2 Strassen rank-7 decomposition
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    n_blocks = n // 2
    m_blocks = m // 2
    p_blocks = p // 2
    
    total_rank = n_blocks * m_blocks * p_blocks * 7
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                for c in range(7):
                    col = col_offset + c
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U[i * m + j][col] = S_U[di * 2 + dj][c]
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V[j * p + k][col] = S_V[dj * 2 + dk][c]
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W[k * n + i][col] = S_W[dk * 2 + di][c]
                col_offset += 7
    
    # Try to reduce rank by checking if any column is redundant
    # Build the tensor and check linear dependence
    import numpy as np
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can check if removing a column still gives the correct tensor
    
    def verify_decomp(U_list, V_list, W_list):
        """Check if decomposition is valid."""
        Ua = np.array(U_list, dtype=float)
        Va = np.array(V_list, dtype=float)
        Wa = np.array(W_list, dtype=float)
        rank = Ua.shape[1]
        
        # Compute tensor from decomposition
        # T[i,j,k] should equal sum over r of U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # For matrix mult: (AB)[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = A[i,j]*B[j,k] contributes to output C[i,k]
        # The tensor is T_{i,j,k} = e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
        # Actually for matrix multiplication tensor:
        # T[i,j,k] = 1 if this term contributes to C[i,k] via A[i,j]*B[j,k]
        # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k) for the MM tensor
        
        # The MM tensor T[i,j,k] = 1 for all valid (i,j,k), 0 otherwise
        # Wait no. The MM tensor is defined such that:
        # (A*B)[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k]
        # This should equal sum_j A[i,j]*B[j,k]
        
        # So we need: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta_{j,j} = 1 for the specific j
        # Actually: for fixed i,k, we need sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j]*B[j,k] = sum_j A[i,j]*B[j,k]
        # This requires: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # And for different j, the contributions should separate properly.
        
        # More precisely: the tensor T[i,j,k] = 1 for all i,j,k in range
        # The decomposition gives: T_decomp[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # We need T_decomp[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        # Compute T_decomp
        T_decomp = np.zeros((n, m, p))
        for r in range(rank):
            u_col = Ua[:, r]
            v_col = Va[:, r]
            w_col = Wa[:, r]
            # u_col[i*m+j], v_col[j*p+k], w_col[k*n+i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i,j,k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        
        # Check if all entries are 1
        return np.allclose(T_decomp, 1.0, atol=1e-9)
    
    # Verify current decomposition
    if not verify_decomp(U, V, W):
        # Return as-is if verification fails (shouldn't happen)
        return {"U": U, "V": V, "W": W}
    
    # Try to remove columns one by one
    # For each column r, check if removing it still gives a valid decomposition
    # This is expensive, so let's be smart about it
    
    # Actually, let's try a different approach: use the structure to find redundancies
    # The 12 blocks each contribute 7 columns. If two blocks have identical U, V, W patterns
    # (up to scaling), we might be able to merge.
    
    # For now, return the current 84-rank solution
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
