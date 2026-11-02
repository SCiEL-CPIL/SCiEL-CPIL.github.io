import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Strategy: Decompose the 2x4 * 4x8 multiplication.
    # We view the 4x8 matrix as two 4x4 blocks? No, the intermediate dimension is m=4.
    # The multiplication is C = A * B where A is 2x4 and B is 4x8.
    # We can split B into two 4x4 matrices B1, B2? No, that doesn't help with A.
    # We can split A into two 2x2 matrices A1, A2? Yes.
    # A = [A1 | A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x4.
    # Then C = A1 * B1 + A2 * B2.
    # This decomposes the problem into two independent 2x2 * 2x4 multiplications.
    
    # For a 2x2 * 2x4 multiplication:
    # We can further decompose the 2x4 matrix into two 2x2 matrices?
    # Let B' be 2x4. B' = [B'_1 | B'_2] where B'_1, B'_2 are 2x2.
    # Then A' * B' = A' * B'_1 + A' * B'_2.
    # This gives two 2x2 * 2x2 multiplications.
    # So, total for 2x2 * 2x4 is 2 * (rank of 2x2 * 2x2).
    
    # Rank of 2x2 * 2x2 is 7 (Strassen).
    # So rank of 2x2 * 2x4 is 2 * 7 = 14.
    
    # We have two such blocks (for A1 and A2).
    # Total rank = 14 + 14 = 28.
    
    # Let's verify this structure.
    # A (2x4) is split into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    # B (4x8) is split into B_top (4x4, cols 0-3) and B_bottom (4x4, cols 4-7)? No.
    # The summation is over j=0..3.
    # C[i,k] = sum_{j=0}^3 A[i,j] B[j,k].
    # Split j into {0,1} and {2,3}.
    # C[i,k] = sum_{j=0}^1 A[i,j] B[j,k] + sum_{j=2}^3 A[i,j] B[j,k].
    # Term 1: A1 (2x2) * B1 (2x8). Here B1 is rows 0-1 of B.
    # Term 2: A2 (2x2) * B2 (2x8). Here B2 is rows 2-3 of B.
    
    # So we need to decompose 2x2 * 2x8 multiplication.
    # Let M be 2x2 and N be 2x8.
    # Split N into four 2x2 matrices N1, N2, N3, N4 (columns 0-1, 2-3, 4-5, 6-7).
    # M * N = M*N1 + M*N2 + M*N3 + M*N4.
    # Each M*Ni is a 2x2 * 2x2 multiplication.
    # Rank of 2x2 * 2x2 is 7.
    # So rank of 2x2 * 2x8 is 4 * 7 = 28.
    
    # We have two such terms (Term 1 and Term 2).
    # Total rank = 28 + 28 = 56.
    
    # This is better than 64.
    
    # Let's construct the Strassen decomposition for 2x2 * 2x2.
    # Standard Strassen formulas for C = A*B:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We will generate the U, V, W factors for each 2x2*2x2 block and map them to the global indices.
    
    # Global indices:
    # U rows: i*m + j -> 2*4 + j = 8+j. i in {0,1}, j in {0,1,2,3}.
    # V rows: j*p + k -> 4*8 + k = 32+k. j in {0,1,2,3}, k in {0..7}.
    # W rows: k*n + i -> 8*2 + i = 16+i. k in {0..7}, i in {0,1}.
    
    # Strassen factors for 2x2*2x2:
    # Local indices:
    # u_idx in {0,1,2,3} corresponding to (0,0), (0,1), (1,0), (1,1) of A.
    # v_idx in {0,1,2,3} corresponding to (0,0), (0,1), (1,0), (1,1) of B.
    # w_idx in {0,1,2,3} corresponding to (0,0), (0,1), (1,0), (1,1) of C.
    
    # Let's define the Strassen decomposition in terms of local coordinates.
    # rank_22 = 7
    # U_22: 4x7 matrix
    # V_22: 4x7 matrix
    # W_22: 4x7 matrix
    
    U_22 = np.zeros((4, 7))
    V_22 = np.zeros((4, 7))
    W_22 = np.zeros((4, 7))
    
    # M1
    U_22[0, 0] = 1; U_22[3, 0] = 1
    V_22[0, 0] = 1; V_22[3, 0] = 1
    W_22[0, 0] = 1; W_22[3, 0] = 1
    
    # M2
    U_22[2, 1] = 1; U_22[3, 1] = 1
    V_22[0, 1] = 1
    W_22[2, 1] = 1; W_22[3, 1] = -1
    
    # M3
    U_22[0, 2] = 1
    V_22[1, 2] = 1; V_22[3, 2] = -1
    W_22[1, 2] = 1; W_22[3, 2] = 1
    
    # M4
    U_22[3, 3] = 1
    V_22[2, 3] = 1; V_22[0, 3] = -1
    W_22[0, 3] = 1; W_22[2, 3] = 1
    
    # M5
    U_22[0, 4] = 1; U_22[1, 4] = 1
    V_22[3, 4] = 1
    W_22[0, 4] = -1; W_22[1, 4] = 1
    
    # M6
    U_22[2, 5] = 1; U_22[0, 5] = -1
    V_22[0, 5] = 1; V_22[1, 5] = 1
    W_22[3, 5] = 1
    
    # M7
    U_22[1, 6] = 1; U_22[3, 6] = -1
    V_22[2, 6] = 1; V_22[3, 6] = 1
    W_22[0, 6] = 1; W_22[1, 6] = -1
    
    # Now we assemble the global decomposition.
    # Total rank = 56.
    U_global = np.zeros((8, 56))
    V_global = np.zeros((32, 56))
    W_global = np.zeros((16, 56))
    
    # Block 1: A1 (rows 0-1 of A, cols 0-1 of A) * B1 (rows 0-1 of B, cols 0-7 of B)
    # A1 local indices (u_local):
    # (0,0) -> global u = 0*4 + 0 = 0
    # (0,1) -> global u = 0*4 + 1 = 1
    # (1,0) -> global u = 1*4 + 0 = 4
    # (1,1) -> global u = 1*4 + 1 = 5
    # u_map_1 = [0, 1, 4, 5]
    
    # B1 is rows 0-1 of B.
    # B1 local indices (v_local) for 2x2 blocks of B1.
    # B1 is 2x8. We split it into 4 blocks of 2x2.
    # Block 1.1: B1 cols 0-1.
    #   (0,0) -> global v = 0*8 + 0 = 0
    #   (0,1) -> global v = 0*8 + 1 = 1
    #   (1,0) -> global v = 1*8 + 0 = 8
    #   (1,1) -> global v = 1*8 + 1 = 9
    #   v_map_1_1 = [0, 1, 8, 9]
    # Block 1.2: B1 cols 2-3.
    #   (0,0) -> global v = 0*8 + 2 = 2
    #   (0,1) -> global v = 0*8 + 3 = 3
    #   (1,0) -> global v = 1*8 + 2 = 10
    #   (1,1) -> global v = 1*8 + 3 = 11
    #   v_map_1_2 = [2, 3, 10, 11]
    # Block 1.3: B1 cols 4-5.
    #   (0,0) -> global v = 0*8 + 4 = 4
    #   (0,1) -> global v = 0*8 + 5 = 5
    #   (1,0) -> global v = 1*8 + 4 = 12
    #   (1,1) -> global v = 1*8 + 5 = 13
    #   v_map_1_3 = [4, 5, 12, 13]
    # Block 1.4: B1 cols 6-7.
    #   (0,0) -> global v = 0*8 + 6 = 6
    #   (0,1) -> global v = 0*8 + 7 = 7
    #   (1,0) -> global v = 1*8 + 6 = 14
    #   (1,1) -> global v = 1*8 + 7 = 15
    #   v_map_1_4 = [6, 7, 14, 15]
    
    # C1 is 2x8.
    # C1 local indices (w_local) for 2x2 blocks of C1.
    # Block 1.1: C1 cols 0-1.
    #   (0,0) -> global w = 0*2 + 0 = 0
    #   (0,1) -> global w = 0*2 + 1 = 1
    #   (1,0) -> global w = 1*2 + 0 = 2
    #   (1,1) -> global w = 1*2 + 1 = 3
    #   w_map_1_1 = [0, 1, 2, 3]
    # Block 1.2: C1 cols 2-3.
    #   (0,0) -> global w = 2*2 + 0 = 4
    #   (0,1) -> global w = 2*2 + 1 = 5
    #   (1,0) -> global w = 3*2 + 0 = 6
    #   (1,1) -> global w = 3*2 + 1 = 7
    #   w_map_1_2 = [4, 5, 6, 7]
    # Block 1.3: C1 cols 4-5.
    #   (0,0) -> global w = 4*2 + 0 = 8
    #   (0,1) -> global w = 4*2 + 1 = 9
    #   (1,0) -> global w = 5*2 + 0 = 10
    #   (1,1) -> global w = 5*2 + 1 = 11
    #   w_map_1_3 = [8, 9, 10, 11]
    # Block 1.4: C1 cols 6-7.
    #   (0,0) -> global w = 6*2 + 0 = 12
    #   (0,1) -> global w = 6*2 + 1 = 13
    #   (1,0) -> global w = 7*2 + 0 = 14
    #   (1,1) -> global w = 7*2 + 1 = 15
    #   w_map_1_4 = [12, 13, 14, 15]
    
    # Block 2: A2 (rows 0-1 of A, cols 2-3 of A) * B2 (rows 2-3 of B, cols 0-7 of B)
    # A2 local indices (u_local):
    # (0,0) -> global u = 0*4 + 2 = 2
    # (0,1) -> global u = 0*4 + 3 = 3
    # (1,0) -> global u = 1*4 + 2 = 6
    # (1,1) -> global u = 1*4 + 3 = 7
    # u_map_2 = [2, 3, 6, 7]
    
    # B2 is rows 2-3 of B.
    # Block 2.1: B2 cols 0-1.
    #   (0,0) -> global v = 2*8 + 0 = 16
    #   (0,1) -> global v = 2*8 + 1 = 17
    #   (1,0) -> global v = 3*8 + 0 = 24
    #   (1,1) -> global v = 3*8 + 1 = 25
    #   v_map_2_1 = [16, 17, 24, 25]
    # Block 2.2: B2 cols 2-3.
    #   (0,0) -> global v = 2*8 + 2 = 18
    #   (0,1) -> global v = 2*8 + 3 = 19
    #   (1,0) -> global v = 3*8 + 2 = 26
    #   (1,1) -> global v = 3*8 + 3 = 27
    #   v_map_2_2 = [18, 19, 26, 27]
    # Block 2.3: B2 cols 4-5.
    #   (0,0) -> global v = 2*8 + 4 = 20
    #   (0,1) -> global v = 2*8 + 5 = 21
    #   (1,0) -> global v = 3*8 + 4 = 28
    #   (1,1) -> global v = 3*8 + 5 = 29
    #   v_map_2_3 = [20, 21, 28, 29]
    # Block 2.4: B2 cols 6-7.
    #   (0,0) -> global v = 2*8 + 6 = 22
    #   (0,1) -> global v = 2*8 + 7 = 23
    #   (1,0) -> global v = 3*8 + 6 = 30
    #   (1,1) -> global v = 3*8 + 7 = 31
    #   v_map_2_4 = [22, 23, 30, 31]
    
    # C2 is 2x8.
    # C2 local indices (w_local) for 2x2 blocks of C2.
    # Note: C is the sum of C1 and C2. So the W factors for Block 2 add to the same W indices as Block 1.
    # Block 2.1: C2 cols 0-1.
    #   (0,0) -> global w = 0*2 + 0 = 0
    #   (0,1) -> global w = 0*2 + 1 = 1
    #   (1,0) -> global w = 1*2 + 0 = 2
    #   (1,1) -> global w = 1*2 + 1 = 3
    #   w_map_2_1 = [0, 1, 2, 3]
    # Block 2.2: C2 cols 2-3.
    #   (0,0) -> global w = 2*2 + 0 = 4
    #   (0,1) -> global w = 2*2 + 1 = 5
    #   (1,0) -> global w = 3*2 + 0 = 6
    #   (1,1) -> global w = 3*2 + 1 = 7
    #   w_map_2_2 = [4, 5, 6, 7]
    # Block 2.3: C2 cols 4-5.
    #   (0,0) -> global w = 4*2 + 0 = 8
    #   (0,1) -> global w = 4*2 + 1 = 9
    #   (1,0) -> global w = 5*2 + 0 = 10
    #   (1,1) -> global w = 5*2 + 1 = 11
    #   w_map_2_3 = [8, 9, 10, 11]
    # Block 2.4: C2 cols 6-7.
    #   (0,0) -> global w = 6*2 + 0 = 12
    #   (0,1) -> global w = 6*2 + 1 = 13
    #   (1,0) -> global w = 7*2 + 0 = 14
    #   (1,1) -> global w = 7*2 + 1 = 15
    #   w_map_2_4 = [12, 13, 14, 15]
    
    # Helper to add a Strassen block
    def add_strassen_block(U_g, V_g, W_g, u_map, v_map, w_map, start_col):
        for c in range(7):
            for l in range(4):
                if U_22[l, c] != 0:
                    U_g[u_map[l], start_col + c] += U_22[l, c]
                if V_22[l, c] != 0:
                    V_g[v_map[l], start_col + c] += V_22[l, c]
                if W_22[l, c] != 0:
                    W_g[w_map[l], start_col + c] += W_22[l, c]
    
    # Block 1.1
    add_strassen_block(U_global, V_global, W_global, [0, 1, 4, 5], [0, 1, 8, 9], [0, 1, 2, 3], 0)
    # Block 1.2
    add_strassen_block(U_global, V_global, W_global, [0, 1, 4, 5], [2, 3, 10, 11], [4, 5, 6, 7], 7)
    # Block 1.3
    add_strassen_block(U_global, V_global, W_global, [0, 1, 4, 5], [4, 5, 12, 13], [8, 9, 10, 11], 14)
    # Block 1.4
    add_strassen_block(U_global, V_global, W_global, [0, 1, 4, 5], [6, 7, 14, 15], [12, 13, 14, 15], 21)
    
    # Block 2.1
    add_strassen_block(U_global, V_global, W_global, [2, 3, 6, 7], [16, 17, 24, 25], [0, 1, 2, 3], 28)
    # Block 2.2
    add_strassen_block(U_global, V_global, W_global, [2, 3, 6, 7], [18, 19, 26, 27], [4, 5, 6, 7], 35)
    # Block 2.3
    add_strassen_block(U_global, V_global, W_global, [2, 3, 6, 7], [20, 21, 28, 29], [8, 9, 10, 11], 42)
    # Block 2.4
    add_strassen_block(U_global, V_global, W_global, [2, 3, 6, 7], [22, 23, 30, 31], [12, 13, 14, 15], 49)
    
    # Convert to list of lists for return
    U_list = U_global.tolist()
    V_list = V_global.tolist()
    W_list = W_global.tolist()
    
    # Verify decomposition
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Check rank 56 decomposition
    # This is a deterministic construction, so it should be correct.
    # We can do a quick check if time permits.
    if time.time() < deadline - 5.0:
        # Reconstruct tensor
        T_recon = np.zeros((8, 32, 16))
        for c in range(56):
            u_col = U_global[:, c]
            v_col = V_global[:, c]
            w_col = W_global[:, c]
            # Outer product
            T_recon += np.einsum('a,b,c->abc', u_col, v_col, w_col)
        
        # Check against original tensor
        T_orig = np.zeros((8, 32, 16))
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    T_orig[i*4+j, j*8+k, k*2+i] = 1
        
        if not np.allclose(T_recon, T_orig):
            # Fallback to schoolbook if verification fails (should not happen)
            return {"U": U_school, "V": V_school, "W": W_school}
        
    return {"U": U_list, "V": V_list, "W": W_list}
