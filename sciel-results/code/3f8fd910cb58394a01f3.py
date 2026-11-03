import numpy as np
import time

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = 3, 4, 4
    
    def verify(U, V, W):
        """Check exact tensor decomposition T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]"""
        # Build tensor T[i,j,k] = 1 if C[k] gets contribution from A[i]*B[j]
        # C[k] = C[k%n][k//n], i.e., C[row=k%n][col=k//n]
        # C[row][col] = sum_t A[row][t]*B[t][col]
        # So T[i,j,k] = 1 iff (i//m)*(j%m) + (i%m)*(j//m) ... 
        # Actually: A is n x m, indexed by row=i//m, col=i%m
        # B is m x p, indexed by row=j//p, col=j%p
        # C is p x n (trace convention W[k*n+i,r]), k = col*n + row
        # C[row][col] = sum_t A[row][t] * B[t][col]
        # So T[i,j,k] = 1 iff i//m == k%n AND j%p == k//m AND i%m == j//p
        # Wait, let me re-derive. C is p x n matrix. C[r][c] where r in 0..p-1, c in 0..n-1.
        # W index: k = r * n + c (since W has shape p*n, and W[k*n+i,r] in the contract means... 
        # Actually the contract says W[k*n+i,r] which is confusing. Let me re-read.
        # "output factor follows the official trace convention W[k*n+i,r]"
        # W has shape (p*n, rank). So W[row, r] where row = k*n + i? That's p*n rows.
        # Actually W has p*n = 12 rows. The index is just 0..11.
        # C is p x n = 4 x 3 matrix. C[r][c] for r in 0..3, c in 0..2.
        # The trace convention: C[r][c] = sum_k W[(r*n+c), k] * M_k? No...
        # Standard: C = sum_r U[:,r] * V[:,r] outer... no.
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # where i indexes A (n*m=12 elements), j indexes B (m*p=16), k indexes C (p*n=12)
        # A[i] = A[i//m][i%m], B[j] = B[j//p][j%p], C[k] = C[k//n][k%n]
        # C[r][c] = sum_t A[r][t] * B[t][c]
        # So C[k//n][k%n] = sum_t A[k//n][t] * B[t][k%n]
        # T[i,j,k] = 1 iff i//m == k//n AND i%m == j//p AND j%p == k%n
        rank = U.shape[1]
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        
        # Expected tensor
        T_exp = np.zeros((n*m, m*p, p*n))
        for i in range(n*m):
            ar, ac = i // m, i % m
            for j in range(m*p):
                br, bc = j // p, j % p
                if ac == br:
                    # C[ar][bc] gets contribution
                    k = bc * n + ar
                    T_exp[i, j, k] = 1
        
        return np.allclose(T, T_exp, atol=1e-9)
    
    def try_decomposition(rank, U, V, W):
        """Verify and return rank if valid, else None"""
        if time.time() > deadline:
            return None
        if verify(U, V, W):
            return rank
        return None
    
    # ===== Current known-good solution: rank 44 =====
    U44 = np.zeros((n * m, 44))
    V44 = np.zeros((m * p, 44))
    W44 = np.zeros((p * n, 44))
    
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra part (cols 7-10) for Row 2 of A
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
    
    def add_block(U, V, W, block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
    
    add_block(U44, V44, W44, 0, 0, 0, 0, 0, 0, 0)
    add_block(U44, V44, W44, 1, 0, 2, 2, 0, 0, 0)
    add_block(U44, V44, W44, 2, 0, 0, 0, 2, 0, 2)
    add_block(U44, V44, W44, 3, 0, 2, 2, 2, 0, 2)
    
    # Verify rank 44
    r44 = try_decomposition(44, U44, V44, W44)
    if r44 is None:
        # Fallback: return zeros (shouldn't happen)
        U_f = np.zeros((n*m, 48))
        V_f = np.zeros((m*p, 48))
        W_f = np.zeros((p*n, 48))
        for i in range(n*m):
            for j in range(m*p):
                if (i % m) == (j // p):
                    k = (j % p) * n + (i // m)
                    r = i * (m*p) + j
                    U_f[i, r] = 1
                    V_f[j, r] = 1
                    W_f[k, r] = 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    best_rank = 44
    best_U, best_V, best_W = U44, V44, W44
    
    # ===== Attempt: rank 40 by sharing trivial row-2 factors =====
    # Key insight: The trivial factors for row 2 (a20*b00, a21*b10, a20*b01, a21*b11)
    # have U depending only on a_row=2 and a_col, V depending only on b_row and b_col,
    # W depending only on c_row=2 and c_col.
    # 
    # In the 4-block tiling, blocks 0 and 1 both have c_row=0, and blocks 2,3 have c_row=0.
    # Wait, all blocks have c_row_start=0. So C rows are all 0-2.
    # Block 0: A cols 0,1 * B rows 0,1 cols 0,1 -> C rows 0,1,2 cols 0,1
    # Block 1: A cols 2,3 * B rows 2,3 cols 0,1 -> C rows 0,1,2 cols 0,1
    # Block 2: A cols 0,1 * B rows 0,1 cols 2,3 -> C rows 0,1,2 cols 2,3
    # Block 3: A cols 2,3 * B rows 2,3 cols 2,3 -> C rows 0,1,2 cols 2,3
    #
    # The trivial row-2 part in each block:
    # Block 0: a20*b00->c20, a21*b10->c20, a20*b01->c21, a21*b11->c21
    # Block 1: a22*b20->c20, a23*b30->c20, a22*b21->c21, a23*b31->c21
    # Block 2: a20*b02->c22, a21*b12->c22, a20*b03->c23, a21*b13->c23
    # Block 3: a22*b22->c22, a23*b32->c22, a22*b23->c23, a23*b33->c23
    #
    # These are all distinct (different A indices, B indices, C indices), so no sharing possible
    # at the individual factor level. However, we can try to combine Strassen + trivial more cleverly.
    #
    # Alternative: use a 3x4x4 direct approach with rank minimization.
    # The tensor has 12*16*12 = 2304 entries, 135 of which are 1.
    # Rank lower bound: max(n*m, m*p, p*n) = 16. Upper bound: 48 (naive).
    #
    # Let me try a different tiling: split A into 2 rows + 1 row differently.
    # Use Strassen on 2x2x2 for rows 0,1 of A, and handle row 2 with rank 4.
    # But tile B differently: instead of 2x2 blocks, use 2x4 and 2x4 for B rows.
    #
    # New idea: Split A into rows {0,1} and row {2}.
    # For rows {0,1}: multiply 2x4 by 4x4 to get 2x4.
    #   Split B into cols {0,1} and {2,3}: 
    #     2x4 * 4x2 -> 2x2 (use Strassen: rank 7) for each of 2 B-col-blocks
    #     Total: 7*2 = 14 for the 2x4 by 4x4 part
    # For row {2}: multiply 1x4 by 4x4 to get 1x4.
    #   Split B into cols {0,1} and {2,3}:
    #     1x4 * 4x2 -> 1x2 (rank 4) for each of 2 B-col-blocks
    #     Total: 4*2 = 8 for the 1x4 by 4x4 part
    # Grand total: 14 + 8 = 22? That seems too low. Let me check.
    #
    # Wait, I need to be more careful. Let me reconsider.
    # A is 3x4, B is 4x4, C is 4x3 (in trace convention, W has p*n=12 rows).
    #
    # Actually, let me try a rank-40 approach by being smarter about the tiling.
    # 
    # Alternative tiling: Split A into 2+1 rows, B into 2+2 rows, C into 2+2 cols.
    # Block (0,0): A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen rank 7)
    # Block (0,1): A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (Strassen rank 7)
    # Block (1,0): A[2:3, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (rank 4)
    # Block (1,1): A[2:3, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (rank 4)
    # Block (0,2): A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2] (Strassen rank 7)
    # Block (0,3): A[0:2, 2:4] * B[2:4, 2:4] -> C[2:4, 0:2] (Strassen rank 7)
    # Block (1,2): A[2:3, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2] (rank 4)
    # Block (1,3): A[2:3, 2:4] * B[2:4, 2:4] -> C[2:4, 0:2] (rank 4)
    # Total: 4*7 + 4*4 = 28 + 16 = 44. Same as before.
    #
    # Hmm. Let me try to find a rank-40 or lower decomposition through search.
    # I'll use a greedy approach: start with the rank-44 solution and try to merge
    # pairs of rank-1 terms that can be combined.
    
    if time.time() > deadline - 30:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try: can we find a rank-43 solution by modifying one term?
    # Or try a completely different approach with rank 40.
    
    # Let me try a rank-40 construction:
    # Use Strassen (rank 7) for 2x2x2 blocks, but overlap them.
    # 
    # Actually, let me try: split A into rows {0,1,2} as {0,1} and {2},
    # but use a shared Strassen structure.
    #
    # New approach: 3x4x4 = (2+1)x4x4
    # = 2x4x4 + 1x4x4
    # 2x4x4: split B into 2x2 blocks: 4 blocks of 2x2x2, each rank 7 = 28
    # 1x4x4: 4 blocks of 1x2x2, each rank 4 = 16
    # Total: 44. Same.
    #
    # To beat 44, I need a fundamentally different decomposition.
    # Let me try rank 40 with a different block structure.
    
    # Try: 3x4x4 with B split into 4x2 blocks (2 blocks), A split into 2x2 blocks
    # A[0:2, 0:4] * B[0:4, 0:2] -> C[0:2, 0:2] : 2x4x2
    # A[0:2, 0:4] * B[0:4, 2:4] -> C[2:4, 0:2] : 2x4x2
    # A[2:3, 0:4] * B[0:4, 0:2] -> C[0:2, 0:2] : 1x4x2
    # A[2:3, 0:4] * B[0:4, 2:4] -> C[2:4, 0:2] : 1x4x2
    #
    # 2x4x2: split A cols into 2+2: two 2x2x2 blocks, each rank 7 = 14
    # 1x4x2: split A cols into 2+2: two 1x2x2 blocks, each rank 4 = 8
    # Total: 2*14 + 2*8 = 28 + 16 = 44. Still 44.
    
    # It seems 44 is hard to beat with simple tiling. Let me verify the current solution
    # and return it if we can't find better.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
