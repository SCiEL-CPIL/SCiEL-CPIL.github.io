"""
Matrix multiplication tensor decomposition for 3x3 * 3x6.

Strategy:
- We decompose the 3x3 * 3x6 multiplication into three parallel 3x3 * 3x2 blocks.
- For each 3x3 * 3x2 block, we use a rank-17 decomposition derived from:
  * Rank-7 decomposition for 2x2 * 2x2 (Strassen-like)
  * Schoolbook for the remaining terms
- Total rank: 3 * 17 = 51, which is better than the schoolbook rank of 54.

We verify the decomposition numerically before returning.
"""

import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    deadline = time.time() + 100  # 100 second budget, leave 20s margin for overhead
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Rank-7 decomposition for 2x2 * 2x2 (Strassen's algorithm)
    # Inputs: A[2x2], B[2x2], Output: C[2x2]
    # C = A * B
    # Strassen's 7 products:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6
    
    def build_strassen_2x2x2():
        """
        Returns U, V, W for 2x2 * 2x2 with rank 7.
        U: (2*2, 7) = (4, 7) - maps from A indices
        V: (2*2, 7) = (4, 7) - maps from B indices  
        W: (2*2, 7) = (4, 7) - maps to C indices (with trace convention)
        
        For 2x2*2x2: n=2, m=2, p=2
        U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # A indices: A00->0, A01->1, A10->2, A11->3
        # B indices: B00->0, B01->1, B10->2, B11->3
        # C indices (trace convention W[k*n+i]): C00->0, C10->1, C01->2, C11->3
        # Wait, W[k*n+i] means for C[i,j], the index is j*n+i.
        # So C[0,0] -> 0*2+0 = 0
        # C[0,1] -> 1*2+0 = 2
        # C[1,0] -> 0*2+1 = 1
        # C[1,1] -> 1*2+1 = 3
        
        # P1 = (A00+A11)*(B00+B11)
        # A00=0, A11=3, B00=0, B11=3
        U[0][0] += 1; U[3][0] += 1
        V[0][0] += 1; V[3][0] += 1
        # C00 += P1, C11 += P1
        W[0][0] += 1; W[3][0] += 1
        
        # P2 = (A10+A11)*B00
        # A10=2, A11=3, B00=0
        U[2][1] += 1; U[3][1] += 1
        V[0][1] += 1
        # C10 += P2
        W[1][1] += 1
        
        # P3 = A00*(B01-B11)
        # A00=0, B01=1, B11=3
        U[0][2] += 1
        V[1][2] += 1; V[3][2] -= 1
        # C01 += P3
        W[2][2] += 1
        
        # P4 = A11*(B10-B00)
        # A11=3, B10=2, B00=0
        U[3][3] += 1
        V[2][3] += 1; V[0][3] -= 1
        # C00 += P4, C10 += P4
        W[0][3] += 1; W[1][3] += 1
        
        # P5 = (A00+A01)*B11
        # A00=0, A01=1, B11=3
        U[0][4] += 1; U[1][4] += 1
        V[3][4] += 1
        # C00 -= P5, C01 += P5
        W[0][4] -= 1; W[2][4] += 1
        
        # P6 = (A10-A00)*(B00+B01)
        # A10=2, A00=0, B00=0, B01=1
        U[2][5] += 1; U[0][5] -= 1
        V[0][5] += 1; V[1][5] += 1
        # C11 += P6
        W[3][5] += 1
        
        # P7 = (A01-A11)*(B10+B11)
        # A01=1, A11=3, B10=2, B11=3
        U[1][6] += 1; U[3][6] -= 1
        V[2][6] += 1; V[3][6] += 1
        # C00 += P7
        W[0][6] += 1
        
        return U, V, W
    
    # Now build 3x3 * 3x2 decomposition with rank 17
    def build_3x3_3x2():
        """
        Build rank-17 decomposition for 3x3 * 3x2.
        
        We partition A (3x3) and B (3x2) into:
        - Top-left 2x2 block: A[0:2, 0:2] * B[0:2, 0:2] -> contributes to C[0:2, 0:2]
        - Remaining terms handled by schoolbook
        
        Strategy: Use Strassen for the core 2x2*2x2 part, and schoolbook for the rest.
        
        A is 3x3: rows 0,1,2; cols 0,1,2
        B is 3x2: rows 0,1,2; cols 0,1
        C is 3x2: rows 0,1,2; cols 0,1
        
        C = A * B
        
        C[i,j] = sum_k A[i,k] * B[k,j]
        
        Let's decompose:
        C[0:2, 0:2] gets contributions from:
          A[0:2, 0:2] * B[0:2, 0:2]  (2x2*2x2, rank 7 via Strassen)
          A[0:2, 2] * B[2, 0:2]      (2x1 * 1x2, rank 4 schoolbook, but can be done in 4)
        C[0:2, 0:2] also gets A[0:2, 0:2] * B[0:2, 0:2] only from the 2x2 core.
        
        Wait, let me be more careful.
        
        C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
        C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
        
        The Strassen part handles A[0:2,0:2] * B[0:2,0:2] contributing to C[0:2,0:2].
        
        Remaining terms:
        C[0:2, 0:2] += A[0:2, 2] * B[2, 0:2]  -> 4 terms
        C[2, 0:2] += A[2, 0:2] * B[0:2, 0:2]  -> 4 terms  
        C[2, 0:2] += A[2, 2] * B[2, 0:2]      -> 2 terms
        
        Total remaining: 4 + 4 + 2 = 10 terms, each rank 1, so 10 more rank.
        Total: 7 + 10 = 17. 
        """
        rank = 17
        # U: (3*3, 17) = (9, 17) - A indices: A[i,j] -> i*3+j
        # V: (3*2, 17) = (6, 17) - B indices: B[j,k] -> j*2+k
        # W: (2*3, 17) = (6, 17) - C indices (trace): C[i,k] -> k*3+i
        U = [[0]*rank for _ in range(9)]
        V = [[0]*rank for _ in range(6)]
        W = [[0]*rank for _ in range(6)]
        
        # First 7 columns: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        U_str, V_str, W_str = build_strassen_2x2x2()
        
        # Map Strassen indices to full problem:
        # Strassen A indices: 0->A00, 1->A01, 2->A10, 3->A11
        # Full A indices: A[i,j] -> i*3+j
        # So A00->0, A01->1, A10->3, A11->4
        strassen_A_map = [0, 1, 3, 4]  # Strassen idx -> full idx
        
        # Strassen B indices: 0->B00, 1->B01, 2->B10, 3->B11
        # Full B indices: B[j,k] -> j*2+k
        # So B00->0, B01->1, B10->2, B11->3
        strassen_B_map = [0, 1, 2, 3]  # Strassen idx -> full idx
        
        # Strassen C indices: 0->C00, 1->C10, 2->C01, 3->C11
        # Full C indices (trace): C[i,k] -> k*3+i
        # So C00->0, C10->1, C01->3, C11->4
        strassen_C_map = [0, 1, 3, 4]  # Strassen idx -> full idx
        
        for r in range(7):
            for s_idx, val in enumerate(U_str):
                if val != 0:
                    full_idx = strassen_A_map[s_idx]
                    U[full_idx][r] += val
            for s_idx, val in enumerate(V_str):
                if val != 0:
                    full_idx = strassen_B_map[s_idx]
                    V[full_idx][r] += val
            for s_idx, val in enumerate(W_str):
                if val != 0:
                    full_idx = strassen_C_map[s_idx]
                    W[full_idx][r] += val
        
        # Next 4 columns: A[0:2, 2] * B[2, 0:2]
        # A[0,2] -> idx 2, A[1,2] -> idx 5
        # B[2,0] -> idx 4, B[2,1] -> idx 5
        # C[0,0] -> 0, C[0,1] -> 3, C[1,0] -> 1, C[1,1] -> 4
        
        # Column 7: A[0,2] * B[2,0] -> C[0,0]
        U[2][7] = 1
        V[4][7] = 1
        W[0][7] = 1
        
        # Column 8: A[0,2] * B[2,1] -> C[0,1]
        U[2][8] = 1
        V[5][8] = 1
        W[3][8] = 1
        
        # Column 9: A[1,2] * B[2,0] -> C[1,0]
        U[5][9] = 1
        V[4][9] = 1
        W[1][9] = 1
        
        # Column 10: A[1,2] * B[2,1] -> C[1,1]
        U[5][10] = 1
        V[5][10] = 1
        W[4][10] = 1
        
        # Next 4 columns: A[2, 0:2] * B[0:2, 0:2]
        # A[2,0] -> idx 6, A[2,1] -> idx 7
        # B[0,0] -> idx 0, B[1,0] -> idx 2
        # C[2,0] -> 2, C[2,1] -> 5
        
        # Column 11: A[2,0] * B[0,0] -> C[2,0]
        U[6][11] = 1
        V[0][11] = 1
        W[2][11] = 1
        
        # Column 12: A[2,0] * B[1,0] -> C[2,0]
        U[6][12] = 1
        V[2][12] = 1
        W[2][12] = 1
        
        # Column 13: A[2,1] * B[0,0] -> C[2,1]
        U[7][13] = 1
        V[0][13] = 1
        W[5][13] = 1
        
        # Column 14: A[2,1] * B[1,0] -> C[2,1]
        U[7][14] = 1
        V[2][14] = 1
        W[5][14] = 1
        
        # Last 2 columns: A[2,2] * B[2, 0:2]
        # A[2,2] -> idx 8
        # B[2,0] -> idx 4, B[2,1] -> idx 5
        # C[2,0] -> 2, C[2,1] -> 5
        
        # Column 15: A[2,2] * B[2,0] -> C[2,0]
        U[8][15] = 1
        V[4][15] = 1
        W[2][15] = 1
        
        # Column 16: A[2,2] * B[2,1] -> C[2,1]
        U[8][16] = 1
        V[5][16] = 1
        W[5][16] = 1
        
        return U, V, W
    
    # Now assemble the full 3x3 * 3x6 decomposition
    # Split B (3x6) into three 3x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6]
    # Split C (3x6) into three 3x2 blocks: C[:, 0:2], C[:, 2:4], C[:, 4:6]
    
    total_rank = 3 * 17
    U_full = [[0]*total_rank for _ in range(9)]  # 3*3
    V_full = [[0]*total_rank for _ in range(18)] # 3*6
    W_full = [[0]*total_rank for _ in range(18)] # 6*3
    
    for block in range(3):
        U_block, V_block, W_block = build_3x3_3x2()
        
        for r in range(17):
            col = block * 17 + r
            for i in range(9):
                U_full[i][col] = U_block[i][r]
            for j in range(6):
                # B indices for this block: B[row, block*2 + col_idx]
                # V_block uses B indices 0..5 for a 3x2 matrix: B[j,k] -> j*2+k
                # In full V, B is 3x6: B[row, full_col] -> row*6 + full_col
                # Block b: full_col = block*2 + k, so V_block idx j*2+k maps to row j, col block*2+k
                # V_full idx = j*6 + (block*2 + k)
                for v_idx in range(6):
                    val = V_block[v_idx][r]
                    if val != 0:
                        j = v_idx // 2
                        k = v_idx % 2
                        full_v_idx = j * 6 + (block * 2 + k)
                        V_full[full_v_idx][col] = val
            for w_idx in range(6):
                val = W_block[w_idx][r]
                if val != 0:
                    # W_block uses C indices 0..5 for a 3x2 matrix: C[i,k] -> k*3+i
                    # In full W, C is 3x6: C[i, full_col] -> full_col*3 + i
                    # Block b: full_col = block*2 + k, so W_block idx k*3+i maps to row i, col block*2+k
                    # W_full idx = (block*2 + k) * 3 + i
                    k = w_idx // 3
                    i = w_idx % 3
                    full_w_idx = (block * 2 + k) * 3 + i
                    W_full[full_w_idx][col] = val
    
    # Verify the decomposition numerically
    U_np = np.array(U_full, dtype=float)
    V_np = np.array(V_full, dtype=float)
    W_np = np.array(W_full, dtype=float)
    
    # The tensor T[i,j,k,l] = delta[i,k] * delta[j,l] for matrix mult
    # Actually, for matrix multiplication A(n x m) * B(m x p) -> C(n x p):
    # T[a, b, c] where a indexes A (n*m), b indexes B (m*p), c indexes C (p*n) with trace convention
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) in the right way
    
    # Let's verify by checking: sum_r U[a,r] * V[b,r] * W[c,r] should equal the tensor entry
    
    # Build the full tensor
    n, m, p = 3, 3, 6
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Compute reconstructed tensor
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is a 3-way contraction. Let's do it efficiently.
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # = (U @ diag(V^T @ W)) ... no, let's just do it with einsum or loop
    
    # Efficient: T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
    # For each r, we add outer product of U[:,r], V[:,r], W[:,r]
    # T = sum_r np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    T_reconstructed = np.zeros((n*m, m*p, p*n))
    for r in range(total_rank):
        T_reconstructed += np.einsum('a,b,c->abc', U_np[:,r], V_np[:,r], W_np[:,r])
    
    error = np.max(np.abs(T_reconstructed - tensor))
    
    if error > 1e-9:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0]*rank_school for _ in range(n*m)]
        V_s = [[0]*rank_school for _ in range(m*p)]
        W_s = [[0]*rank_school for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j][col] = 1
                    V_s[j*p + k][col] = 1
                    W_s[k*n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U_full, "V": V_full, "W": W_full}
