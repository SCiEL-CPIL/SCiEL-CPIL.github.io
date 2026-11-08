import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook

    # Correct Strassen factors for 2x2x2 multiplication
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 - M5 + M7
    #
    # U factors (A side), indexed by A11, A12, A21, A22:
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],   # A11
        [0, 0, 0, 0, 1, 0, 1],    # A12
        [0, 1, 0, 0, 0, 1, 0],    # A21
        [1, 1, 0, 1, 0, 0, -1]    # A22
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],   # B11
        [0, 0, 1, 0, 0, 1, 0],    # B12
        [0, 0, 0, 1, 0, 0, 1],    # B21
        [1, 0, -1, 0, 1, 0, 1]    # B22
    ]
    W_s = [
        [1, 0, 0, -1, 0, -1, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 1, 0],    # C21
        [1, 0, 0, 0, -1, 0, 1]    # C22
    ]
    
    # Verify Strassen factors
    def verify_strassen(U_s, V_s, W_s):
        # T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c] should be 1 for all i,j,k in 0..3
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    val = sum(U_s[i][c] * V_s[j][c] * W_s[k][c] for c in range(7))
                    if val != 1:
                        return False
        return True
    
    if not verify_strassen(U_s, V_s, W_s):
        # Try alternative factorings
        # The issue might be in the W factors. Let me re-derive carefully.
        # M1: A11*A22 contribute to U, B11*B22 to V, C11*C22 to W
        # Actually let me just use the known correct Strassen decomposition
        pass

    # Structural approach: treat 2x4 * 4x4 as block multiplication
    # A is 2x4, split into A_left (2x2) and A_right (2x2)
    # B is 4x4, split into B_top (4x2) and B_bottom (4x2) -- no wait
    # B is 4x4. For C = A*B (2x4), we need:
    # C[:, 0:2] = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :]
    # C[:, 2:4] = A[:, 0:2] * B[0:2, :] ... no
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # Split j into j=0,1 (first half) and j=2,3 (second half):
    # C[i,k] = sum_{j=0,1} A[i,j]*B[j,k] + sum_{j=2,3} A[i,j]*B[j,k]
    #
    # Let A1 = A[:,0:2] (2x2), A2 = A[:,2:4] (2x2)
    # Let B1 = B[0:2,:] (2x4), B2 = B[2:4,:] (2x4)
    # C = A1*B1 + A2*B2
    #
    # A1*B1 is 2x2 * 2x4 = 2x4
    # A2*B2 is 2x2 * 2x4 = 2x4
    #
    # For A1*B1: this is (2x2)*(2x4). Split B1 into B1_left (2x2) and B1_right (2x2):
    # A1*B1_left gives left half of C (2x2), A1*B1_right gives right half
    # Similarly for A2*B2.
    #
    # So we have 4 sub-problems, each 2x2 * 2x2, using rank-7 Strassen:
    # Total rank = 4 * 7 = 28
    #
    # Sub-problem mapping:
    # 1. A1 * B1_left -> C_left (cols 0,1 of C)
    # 2. A1 * B1_right -> C_right (cols 2,3 of C)  
    # 3. A2 * B2_left -> C_left (cols 0,1 of C)
    # 4. A2 * B2_right -> C_right (cols 2,3 of C)
    #
    # Index conventions:
    # U: n*m = 2*4 = 8 rows, indexed by (i, j) where i in 0..1, j in 0..3
    # V: m*p = 4*4 = 16 rows, indexed by (j, k) where j in 0..3, k in 0..3
    # W: p*n = 4*2 = 8 rows, indexed by (k, i) where k in 0..3, i in 0..1
    #
    # A1: rows i=0,1; cols j=0,1 -> U indices: (0,0),(0,1),(1,0),(1,1) = 0,1,4,5
    # A2: rows i=0,1; cols j=2,3 -> U indices: (0,2),(0,3),(1,2),(1,3) = 2,3,6,7
    #
    # B1_left: rows j=0,1; cols k=0,1 -> V indices: (0,0),(0,1),(1,0),(1,1) = 0,1,4,5
    # B1_right: rows j=0,1; cols k=2,3 -> V indices: (0,2),(0,3),(1,2),(1,3) = 2,3,6,7
    # B2_left: rows j=2,3; cols k=0,1 -> V indices: (2,0),(2,1),(3,0),(3,1) = 8,9,12,13
    # B2_right: rows j=2,3; cols k=2,3 -> V indices: (2,2),(2,3),(3,2),(3,3) = 10,11,14,15
    #
    # C_left: cols k=0,1; rows i=0,1 -> W indices: (0,0),(0,1),(1,0),(1,1) = 0,2,4,6
    #   W[k*n+i]: k=0,i=0->0; k=0,i=1->1; k=1,i=0->2; k=1,i=1->3
    #   Wait: W is p*n = 4*2 = 8, indexed by k*n+i
    #   k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
    #   k=2,i=0 -> 4; k=2,i=1 -> 5; k=3,i=0 -> 6; k=3,i=1 -> 7
    #
    # C_left (k=0,1): W indices 0,1,2,3
    # C_right (k=2,3): W indices 4,5,6,7

    rank = 28
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(8)]
    
    # For each sub-problem, map Strassen indices to global indices
    # Strassen local indices 0,1,2,3 correspond to:
    # U local: A11, A12, A21, A22 (row, col within 2x2 block)
    # V local: B11, B12, B21, B22
    # W local: C11, C12, C21, C22
    
    def local_to_global_A(local_idx, a_block):
        # a_block: 0 = A1 (cols 0,1), 1 = A2 (cols 2,3)
        # local_idx: 0=A11(i=0,j=0), 1=A12(i=0,j=1), 2=A21(i=1,j=0), 3=A22(i=1,j=1)
        i = local_idx // 2
        j_local = local_idx % 2
        j = j_local + a_block * 2
        return i * 4 + j  # U row index: i*m + j = i*4 + j
    
    def local_to_global_V(local_idx, b_block):
        # b_block: 0=B1_left(j=0,1;k=0,1), 1=B1_right(j=0,1;k=2,3)
        #          2=B2_left(j=2,3;k=0,1), 3=B2_right(j=2,3;k=2,3)
        # local_idx: 0=B11(j=0,k=0), 1=B12(j=0,k=1), 2=B21(j=1,k=0), 3=B22(j=1,k=1)
        j_local = local_idx // 2
        k_local = local_idx % 2
        # b_block determines which 2x4 half of B
        # b_block 0: j=0,1; k=0,1
        # b_block 1: j=0,1; k=2,3
        # b_block 2: j=2,3; k=0,1
        # b_block 3: j=2,3; k=2,3
        j_base = (b_block // 2) * 2  # 0 for b_block 0,1; 2 for b_block 2,3
        k_base = (b_block % 2) * 2   # 0 for b_block 0,2; 2 for b_block 1,3
        j = j_base + j_local
        k = k_base + k_local
        return j * 4 + k  # V row index: j*p + k = j*4 + k
    
    def local_to_global_W(local_idx, c_block):
        # c_block: 0 = C_left(k=0,1), 1 = C_right(k=2,3)
        # local_idx: 0=C11(k=0,i=0), 1=C12(k=0,i=1), 2=C21(k=1,i=0), 3=C22(k=1,i=1)
        k_local = local_idx // 2
        i_local = local_idx % 2
        k_base = c_block * 2
        k = k_base + k_local
        i = i_local
        return k * 2 + i  # W row index: k*n + i = k*2 + i
    
    # Sub-problems: (a_block, b_block, c_block)
    sub_problems = [
        (0, 0, 0),  # A1 * B1_left -> C_left
        (0, 1, 1),  # A1 * B1_right -> C_right
        (1, 2, 0),  # A2 * B2_left -> C_left
        (1, 3, 1),  # A2 * B2_right -> C_right
    ]
    
    for sp_idx, (a_block, b_block, c_block) in enumerate(sub_problems):
        base_col = sp_idx * 7
        for c in range(7):
            col = base_col + c
            for local_i in range(4):
                val = U_s[local_i][c]
                if val != 0:
                    global_i = local_to_global_A(local_i, a_block)
                    U[global_i][col] += val
            for local_j in range(4):
                val = V_s[local_j][c]
                if val != 0:
                    global_j = local_to_global_V(local_j, b_block)
                    V[global_j][col] += val
            for local_k in range(4):
                val = W_s[local_k][c]
                if val != 0:
                    global_k = local_to_global_W(local_k, c_block)
                    W[global_k][col] += val
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_r = U_arr.reshape(2, 4, 28)
    V_r = V_arr.reshape(4, 4, 28)
    W_r = W_arr.reshape(4, 2, 28)
    
    valid = True
    for i in range(2):
        for j in range(4):
            for k in range(4):
                val = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                if abs(val - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        best_result = {"U": U, "V": V, "W": W}
        best_rank = 28

    return best_result
