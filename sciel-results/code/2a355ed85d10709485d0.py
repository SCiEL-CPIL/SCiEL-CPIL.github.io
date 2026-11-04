"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build rank-49 decomposition for 4x4x4 using Strassen at two levels
    # Then extend to 4x4x8 with rank 98
    
    # First, define Strassen 2x2x2 decomposition
    # U indices: (i_block, i_sub) where i_block in {0,1}, i_sub in {0,1} -> 4 total
    # V indices: (j_block, j_sub) -> 4 total  
    # W indices: (k_block, k_sub) -> 4 total
    # For 2x2x2: U[i,j] for i,j in 0..1, V[j,k], W[k,i]
    
    # Strassen 2x2x2: 7 rank-1 terms
    # Term t: U_t[i,j], V_t[j,k], W_t[k,i]
    # M1: (A11+A22)(B11+B22) -> U: [[1,0],[0,1]], V: [[1,0],[0,1]], W: [[1,0],[0,1]]
    # M2: (A21+A22)B11 -> U: [[0,0],[1,1]], V: [[1,0],[0,0]], W: [[1,0],[0,0]]
    # M3: A11(B21-B22) -> U: [[1,0],[0,0]], V: [[0,0],[1,-1]], W: [[0,1],[0,0]]
    # M4: A22(B12-B11) -> U: [[0,0],[0,1]], V: [[-1,1],[0,0]], W: [[0,0],[1,0]]
    # M5: (A11+A12)B22 -> U: [[1,1],[0,0]], V: [[0,0],[0,1]], W: [[0,0],[0,1]]
    # M6: (A21-A11)(B11+B12) -> U: [[-1,0],[1,0]], V: [[1,1],[0,0]], W: [[0,0],[0,1]]
    # M7: (A12-A22)(B21+B22) -> U: [[0,1],[0,-1]], V: [[0,0],[1,1]], W: [[1,0],[0,0]]
    
    def strassen_2x2():
        """Return U, V, W for 2x2x2 Strassen, each 2x2x7 (7 rank terms)"""
        U = []
        V = []
        W = []
        # Each term is a 2x2 matrix for U, V, W
        terms = [
            # M1
            ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
            # M2
            ([[0,0],[1,1]], [[1,0],[0,0]], [[1,0],[0,0]]),
            # M3
            ([[1,0],[0,0]], [[0,0],[1,-1]], [[0,1],[0,0]]),
            # M4
            ([[0,0],[0,1]], [[-1,1],[0,0]], [[0,0],[1,0]]),
            # M5
            ([[1,1],[0,0]], [[0,0],[0,1]], [[0,0],[0,1]]),
            # M6
            ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
            # M7
            ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
        ]
        for ut, vt, wt in terms:
            U.append(ut)
            V.append(vt)
            W.append(wt)
        return U, V, W
    
    # For 4x4x4: we have 2x2 blocks of 2x2 matrices
    # Block indices: (bi, bj, bk) where bi,bj,bk in {0,1}
    # A has blocks A[bi,bj] (2x2 each), B has blocks B[bj,bk] (2x2 each)
    # C has blocks C[bi,bk] (2x2 each)
    
    # Apply Strassen to the 2x2 grid of blocks:
    # The "super" Strassen uses the 7 terms, each involving 2x2 block multiplications
    # Each 2x2 block multiplication uses the 2x2x2 Strassen (7 terms)
    # Total: 7 * 7 = 49 rank-1 terms
    
    U_2x2, V_2x2, W_2x2 = strassen_2x2()
    
    # Now build 4x4x4 decomposition
    # U_4x4: 4x49, V_4x4: 4x49, W_4x4: 4x49
    # For super-term s (0..6) and sub-term t (0..6):
    # U_4x4[i, s*7+t] = U_2x2[s][bi][bj] * U_2x2[t][i_sub][j_sub]
    # where i = bi*2 + i_sub, j = bj*2 + j_sub
    # V_4x4[j, s*7+t] = V_2x2[s][bj][bk] * V_2x2[t][j_sub][k_sub]
    # W_4x4[k, s*7+t] = W_2x2[s][bk][bi] * W_2x2[t][k_sub][i_sub]
    
    U_4x4 = [[0]*49 for _ in range(4)]
    V_4x4 = [[0]*49 for _ in range(4)]
    W_4x4 = [[0]*49 for _ in range(4)]
    
    for s in range(7):
        for t in range(7):
            col = s*7 + t
            # U_2x2[s] is 2x2, U_2x2[t] is 2x2
            for bi in range(2):
                for i_sub in range(2):
                    i = bi*2 + i_sub
                    for bj in range(2):
                        for j_sub in range(2):
                            j = bj*2 + j_sub
                            U_4x4[i][col] += U_2x2[s][bi][bj] * U_2x2[t][i_sub][j_sub]
                        
                        for bk in range(2):
                            for k_sub in range(2):
                                k = bk*2 + k_sub
                                V_4x4[j][col] += V_2x2[s][bj][bk] * V_2x2[t][j_sub][k_sub]
                                W_4x4[k][col] += W_2x2[s][bk][bi] * W_2x2[t][k_sub][i_sub]
    
    # Now extend to 4x4x8: B is 4x8, so B has two 4x4 blocks: B[:, 0:4] and B[:, 4:8]
    # C = A @ B = [A @ B1, A @ B2]
    # Use the 4x4x4 decomposition for each half
    # U_4x8 = [U_4x4, U_4x4] (repeat for both halves)
    # V_4x8: for first half, V_4x4; for second half, V_4x4
    # W_4x8: W for first half is W_4x4, for second half is W_4x4
    # But we need to map the output indices correctly
    
    # Actually, for 4x4x8:
    # U: 4x4 (n*m = 16 rows) x rank
    # V: 4x8 (m*p = 32 rows) x rank  
    # W: 8x4 (p*n = 32 rows) x rank
    
    # Wait, let me re-read the interface:
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (32, rank)
    # W: (p*n, rank) = (32, rank)
    
    # For 4x4x4: U is (4*4, rank) = (16, rank), V is (4*4, rank) = (16, rank), W is (4*4, rank) = (16, rank)
    # For 4x4x8: U is (4*4, rank) = (16, rank), V is (4*8, rank) = (32, rank), W is (8*4, rank) = (32, rank)
    
    # My U_4x4 above is 4x49, but I need 16x49. Let me fix this.
    # U[i,j] where i is row of A (0-3), j is col of A (0-3), so index i*4+j in 0..15
    # V[j,k] where j is row of B (0-3), k is col of B (0-3), so index j*4+k in 0..15
    # W[k,i] where k is col of B (0-3), i is row of A (0-3), so index k*4+i in 0..15
    
    # Let me rebuild properly
    
    U_4x4 = [[0]*49 for _ in range(16)]  # 16 = 4*4
    V_4x4 = [[0]*49 for _ in range(16)]  # 16 = 4*4
    W_4x4 = [[0]*49 for _ in range(16)]  # 16 = 4*4
    
    for s in range(7):
        for t in range(7):
            col = s*7 + t
            for bi in range(2):
                for bj in range(2):
                    for bk in range(2):
                        for i_sub in range(2):
                            for j_sub in range(2):
                                i = bi*2 + i_sub
                                j = bj*2 + j_sub
                                U_4x4[i*4+j][col] += U_2x2[s][bi][bj] * U_2x2[t][i_sub][j_sub]
                                
                                for k_sub in range(2):
                                    k = bk*2 + k_sub
                                    V_4x4[j*4+k][col] += V_2x2[s][bj][bk] * V_2x2[t][j_sub][k_sub]
                                    W_4x4[k*4+i][col] += W_2x2[s][bk][bi] * W_2x2[t][k_sub][i_sub]
    
    # Verify 4x4x4 decomposition
    # T[i,j,k] = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    # Should equal 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    
    # Now for 4x4x8:
    # U: 16 x rank
    # V: 32 x rank (j in 0..3, k in 0..7)
    # W: 32 x rank (k in 0..7, i in 0..3)
    
    # B has two 4x4 blocks: B[:, 0:4] and B[:, 4:8]
    # For the first block (k in 0..3): use V_4x4[j*4+k, r] and W_4x4[k*4+i, r]
    # For the second block (k in 4..7): use V_4x4[j*4+(k-4), r] and W_4x4[(k-4)*4+i, r]
    
    # So we can reuse the same U, and for V and W, just repeat the pattern
    # Rank = 49 * 2 = 98
    
    rank = 98
    U_4x8 = [[0]*rank for _ in range(16)]
    V_4x8 = [[0]*rank for _ in range(32)]
    W_4x8 = [[0]*rank for _ in range(32)]
    
    for r in range(49):
        # First block (k in 0..3)
        for i in range(4):
            for j in range(4):
                U_4x8[i*4+j][r] = U_4x4[i*4+j][r]
        for j in range(4):
            for k in range(4):
                V_4x8[j*4+k][r] = V_4x4[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W_4x8[k*4+i][r] = W_4x4[k*4+i][r]
        
        # Second block (k in 4..7)
        for i in range(4):
            for j in range(4):
                U_4x8[i*4+j][r+49] = U_4x4[i*4+j][r]
        for j in range(4):
            for k in range(4):
                V_4x8[j*4+(k+4)][r+49] = V_4x4[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W_4x8[(k+4)*4+i][r+49] = W_4x4[k*4+i][r]
    
    # Verify the decomposition
    # T[i,j,k] = sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*4+i, r]
    # Should equal 1 for all i,j,k
    
    # Quick verification
    import numpy as np
    U_np = np.array(U_4x8)
    V_np = np.array(V_4x8)
    W_np = np.array(W_4x8)
    
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(8):
                val = sum(U_np[i*4+j, r] * V_np[j*8+k, r] * W_np[k*4+i, r] for r in range(rank))
                if abs(val - 1) > max_err:
                    max_err = abs(val - 1)
    
    # If verification fails, fall back to schoolbook
    if max_err > 1e-9:
        # Schoolbook decomposition
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
    
    return {"U": U_4x8, "V": V_4x8, "W": W_4x8}
# EVOLVE-BLOCK-END
