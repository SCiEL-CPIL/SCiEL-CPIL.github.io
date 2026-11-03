import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def check_decomposition(U, V, W, n, m, p):
    """
    Check if U, V, W form a valid decomposition.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Condition: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k
    """
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0
                for r in range(R):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # We will build a decomposition using block decomposition.
    # Strategy:
    # 1. Split m=4 into m1=2, m2=2.
    #    AB = A[:, :2] B[:2, :] + A[:, 2:] B[2:, :]
    #    Two blocks of size 3 x 2 x 5.
    #
    # 2. For each 3 x 2 x 5 block:
    #    Split n=3 into n1=2, n2=1.
    #    AB = A[:2, :] B + A[2:, :] B
    #    Block 1: 2 x 2 x 5
    #    Block 2: 1 x 2 x 5
    #
    # 3. For 2 x 2 x 5 block:
    #    Split p=5 into p1=2, p2=2, p3=1.
    #    AB = A B[:, :2] + A B[:, 2:4] + A B[:, 4:]
    #    Block 1: 2 x 2 x 2 (Strassen, rank 7)
    #    Block 2: 2 x 2 x 2 (Strassen, rank 7)
    #    Block 3: 2 x 2 x 1 (Rank 4)
    #    Total rank: 7 + 7 + 4 = 18
    #
    # 4. For 1 x 2 x 5 block:
    #    Rank = 1 * 2 * 5 = 10 (Trivial)
    #
    # Total rank for one 3x2x5 block: 18 + 10 = 28
    # Total rank for 3x4x5: 28 + 28 = 56

    # Helper: 2x2x2 Strassen
    def strassen_222():
        # Returns U, V, W for 2x2x2
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indices: i in {0,1}, j in {0,1}, k in {0,1}
        # u_idx = i*2 + j
        # v_idx = j*2 + k
        # w_idx = k*2 + i
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # Strassen formulas:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B11 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's map to U, V, W entries
        # r=0: M1
        U[0, 0] = 1; U[3, 0] = 1  # A11, A22
        V[0, 0] = 1; V[3, 0] = 1  # B11, B22 (j=0,k=0 and j=1,k=1)
        W[0, 0] = 1; W[3, 0] = 1  # C11, C22 (k=0,i=0 and k=1,i=1)
        
        # r=1: M2
        U[2, 1] = 1; U[3, 1] = 1  # A21, A22
        V[0, 1] = 1               # B11
        W[2, 1] = 1; W[3, 1] = 1  # C21, C22 (k=0,i=1 and k=1,i=1)
        
        # r=2: M3
        U[0, 2] = 1               # A11
        V[2, 2] = 1; V[3, 2] = -1 # B21, -B22
        W[1, 2] = 1; W[3, 2] = 1  # C12, C22 (k=1,i=0 and k=1,i=1)
        
        # r=3: M4
        U[3, 3] = 1               # A22
        V[0, 3] = 1; V[1, 3] = -1 # B11, -B12
        W[0, 3] = 1; W[2, 3] = 1  # C11, C21 (k=0,i=0 and k=0,i=1)
        
        # r=4: M5
        U[0, 4] = 1; U[1, 4] = 1  # A11, A12
        V[3, 4] = 1               # B22
        W[0, 4] = -1; W[1, 4] = 1 # -C11, C12 (k=0,i=0 and k=1,i=0)
        
        # r=5: M6
        U[2, 5] = 1; U[0, 5] = -1 # A21, -A11
        V[0, 5] = 1; V[1, 5] = 1  # B11, B12
        W[1, 5] = 1; W[3, 5] = 1  # C12, C22 (k=1,i=0 and k=1,i=1)
        
        # r=6: M7
        U[1, 6] = 1; U[3, 6] = -1 # A12, -A22
        V[2, 6] = 1; V[3, 6] = 1  # B21, B22
        W[0, 6] = 1; W[2, 6] = 1  # C11, C21 (k=0,i=0 and k=0,i=1)
        
        return U, V, W

    # Helper: 2x2x1 (Rank 4)
    def trivial_221():
        # U: (4, 4), V: (2, 4), W: (2, 4)
        # i in {0,1}, j in {0,1}, k=0
        # u_idx = i*2+j
        # v_idx = j*2+0 = 2j
        # w_idx = 0*2+i = i
        
        U = np.zeros((4, 4), dtype=int)
        V = np.zeros((2, 4), dtype=int)
        W = np.zeros((2, 4), dtype=int)
        
        col = 0
        for i in range(2):
            for j in range(2):
                U[i*2+j, col] = 1
                V[j*2, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W

    # Helper: 1x2x5 (Rank 10)
    def trivial_125():
        # U: (2, 10), V: (10, 10), W: (5, 10)
        # i=0, j in {0,1}, k in {0..4}
        # u_idx = 0*2+j = j
        # v_idx = j*5+k
        # w_idx = k*1+0 = k
        
        U = np.zeros((2, 10), dtype=int)
        V = np.zeros((10, 10), dtype=int)
        W = np.zeros((5, 10), dtype=int)
        
        col = 0
        for j in range(2):
            for k in range(5):
                U[j, col] = 1
                V[j*5+k, col] = 1
                W[k, col] = 1
                col += 1
        return U, V, W

    # Helper: 2x2x5 (Rank 18)
    def block_225():
        # Combines 2x2x2 (Strassen) and 2x2x1
        # U: (4, 18), V: (10, 18), W: (10, 18)
        # Indices for 2x2x5:
        # i in {0,1}, j in {0,1}, k in {0..4}
        # u_idx = i*2+j
        # v_idx = j*5+k
        # w_idx = k*2+i
        
        U_222, V_222, W_222 = strassen_222()
        U_221, V_221, W_221 = trivial_221()
        
        U = np.zeros((4, 18), dtype=int)
        V = np.zeros((10, 18), dtype=int)
        W = np.zeros((10, 18), dtype=int)
        
        # Part 1: k in {0,1} -> 2x2x2
        # Map 2x2x2 indices to 2x2x5 indices
        # u_idx is same (i*2+j)
        # v_idx: 2x2x2 uses j*2+k (k in 0,1). 2x2x5 uses j*5+k.
        #   j=0: v_idx 2x2x2 is 0,1. 2x2x5 is 0,1. Match.
        #   j=1: v_idx 2x2x2 is 2,3. 2x2x5 is 5,6. Shift by 3? No, 2x2x2 v_idx 2 is j=1,k=0. 2x2x5 v_idx 5 is j=1,k=0.
        #   So for j=0, v_idx is same. For j=1, v_idx +3.
        #   Wait, 2x2x2 v_idx = j*2+k.
        #   j=0: 0,1. 2x2x5 v_idx = 0,1.
        #   j=1: 2,3. 2x2x5 v_idx = 5,6.
        #   So V_222 row 2 -> V row 5. Row 3 -> V row 6.
        # w_idx: 2x2x2 uses k*2+i. 2x2x5 uses k*2+i. Same.
        
        U[:, :7] = U_222
        # V mapping
        V[0, :7] = V_222[0, :]
        V[1, :7] = V_222[1, :]
        V[5, :7] = V_222[2, :]
        V[6, :7] = V_222[3, :]
        W[:, :7] = W_222
        
        # Part 2: k in {2,3} -> 2x2x2 (shifted)
        # k' = k-2.
        # u_idx same.
        # v_idx: j*5 + (k'+2) = j*5 + k' + 2.
        #   j=0: 2,3. 2x2x2 v_idx 0,1 -> +2.
        #   j=1: 7,8. 2x2x2 v_idx 2,3 -> +5.
        # w_idx: (k'+2)*2 + i = k'*2 + i + 4.
        #   2x2x2 w_idx 0..3 -> +4.
        
        U[:, 7:14] = U_222
        V[2, 7:14] = V_222[0, :]
        V[3, 7:14] = V_222[1, :]
        V[7, 7:14] = V_222[2, :]
        V[8, 7:14] = V_222[3, :]
        # W mapping: w_idx 0->4, 1->5, 2->6, 3->7
        W[4, 7:14] = W_222[0, :]
        W[5, 7:14] = W_222[1, :]
        W[6, 7:14] = W_222[2, :]
        W[7, 7:14] = W_222[3, :]
        
        # Part 3: k=4 -> 2x2x1
        # u_idx same.
        # v_idx: j*5+4.
        #   j=0: 4. 2x2x1 v_idx 0 -> +4.
        #   j=1: 9. 2x2x1 v_idx 1 -> +8.
        # w_idx: 4*2+i = 8+i.
        #   i=0: 8. 2x2x1 w_idx 0 -> +8.
        #   i=1: 9. 2x2x1 w_idx 1 -> +8.
        
        U[:, 14:18] = U_221
        V[4, 14:18] = V_221[0, :]
        V[9, 14:18] = V_221[1, :]
        W[8, 14:18] = W_221[0, :]
        W[9, 14:18] = W_221[1, :]
        
        return U, V, W

    # Helper: 3x2x5 (Rank 28)
    def block_325():
        # Combines 2x2x5 (Rank 18) and 1x2x5 (Rank 10)
        # i in {0,1,2}, j in {0,1}, k in {0..4}
        # u_idx = i*2+j
        # v_idx = j*5+k
        # w_idx = k*3+i
        
        U_225, V_225, W_225 = block_225()
        U_125, V_125, W_125 = trivial_125()
        
        U = np.zeros((6, 28), dtype=int)
        V = np.zeros((10, 28), dtype=int)
        W = np.zeros((15, 28), dtype=int)
        
        # Part 1: i in {0,1} -> 2x2x5
        # u_idx: 2x2x5 uses i'*2+j. Here i'=i. Same.
        # v_idx: Same.
        # w_idx: 2x2x5 uses k*2+i'. Here k*3+i.
        #   i=0: 2x2x5 w_idx 0,2 -> 3,6.
        #   i=1: 2x2x5 w_idx 1,3 -> 4,7.
        #   Wait, 2x2x5 w_idx = k*2+i.
        #   k=0: i=0->0, i=1->1. Here: i=0->0, i=1->1.
        #   k=1: i=0->2, i=1->3. Here: i=0->3, i=1->4.
        #   k=2: i=0->4, i=1->5. Here: i=0->6, i=1->7.
        #   k=3: i=0->6, i=1->7. Here: i=0->9, i=1->10.
        #   k=4: i=0->8, i=1->9. Here: i=0->12, i=1->13.
        
        U[:4, :18] = U_225
        V[:, :18] = V_225
        
        # W mapping
        # 2x2x5 W rows 0..9.
        # Row 0 (k=0,i=0) -> W[0]
        # Row 1 (k=0,i=1) -> W[1]
        # Row 2 (k=1,i=0) -> W[3]
        # Row 3 (k=1,i=1) -> W[4]
        # Row 4 (k=2,i=0) -> W[6]
        # Row 5 (k=2,i=1) -> W[7]
        # Row 6 (k=3,i=0) -> W[9]
        # Row 7 (k=3,i=1) -> W[10]
        # Row 8 (k=4,i=0) -> W[12]
        # Row 9 (k=4,i=1) -> W[13]
        
        W[0, :18] = W_225[0, :]
        W[1, :18] = W_225[1, :]
        W[3, :18] = W_225[2, :]
        W[4, :18] = W_225[3, :]
        W[6, :18] = W_225[4, :]
        W[7, :18] = W_225[5, :]
        W[9, :18] = W_225[6, :]
        W[10, :18] = W_225[7, :]
        W[12, :18] = W_225[8, :]
        W[13, :18] = W_225[9, :]
        
        # Part 2: i=2 -> 1x2x5
        # u_idx: 1x2x5 uses j. Here i=2, u_idx = 2*2+j = 4+j.
        #   1x2x5 u_idx 0 -> 4. 1 -> 5.
        # v_idx: Same.
        # w_idx: 1x2x5 uses k. Here k*3+2.
        #   k=0 -> 2.
        #   k=1 -> 5.
        #   k=2 -> 8.
        #   k=3 -> 11.
        #   k=4 -> 14.
        
        U[4, 18:28] = U_125[0, :]
        U[5, 18:28] = U_125[1, :]
        V[:, 18:28] = V_125
        W[2, 18:28] = W_125[0, :]
        W[5, 18:28] = W_125[1, :]
        W[8, 18:28] = W_125[2, :]
        W[11, 18:28] = W_125[3, :]
        W[14, 18:28] = W_125[4, :]
        
        return U, V, W

    # Main: 3x4x5
    # m=4 split into 2,2.
    # Block 1: j in {0,1} -> 3x2x5
    # Block 2: j in {2,3} -> 3x2x5
    
    U_325, V_325, W_325 = block_325()
    
    U = np.zeros((12, 56), dtype=int)
    V = np.zeros((20, 56), dtype=int)
    W = np.zeros((15, 56), dtype=int)
    
    # Block 1: j=0,1
    # u_idx = i*4+j.
    #   3x2x5 u_idx = i*2+j'. j'=j.
    #   i=0: 0,1 -> 0,1.
    #   i=1: 2,3 -> 4,5.
    #   i=2: 4,5 -> 8,9.
    # v_idx = j*5+k.
    #   3x2x5 v_idx = j'*5+k. j'=j. Same.
    # w_idx = k*3+i. Same.
    
    U[0, :28] = U_325[0, :]
    U[1, :28] = U_325[1, :]
    U[4, :28] = U_325[2, :]
    U[5, :28] = U_325[3, :]
    U[8, :28] = U_325[4, :]
    U[9, :28] = U_325[5, :]
    V[:, :28] = V_325
    W[:, :28] = W_325
    
    # Block 2: j=2,3
    # u_idx = i*4+j. j=2,3.
    #   3x2x5 u_idx = i*2+j'. j'=j-2.
    #   i=0: 2,3 -> 2,3.
    #   i=1: 6,7 -> 6,7.
    #   i=2: 10,11 -> 10,11.
    # v_idx = j*5+k. j=2,3.
    #   3x2x5 v_idx = j'*5+k. j'=j-2.
    #   j=2: 2*5+k = 10+k. 3x2x5 v_idx 0..9 (j'=0). Shift +10.
    #   j=3: 3*5+k = 15+k. 3x2x5 v_idx 0..9 (j'=1). Shift +15.
    #   Wait, 3x2x5 v_idx is j'*5+k.
    #   j'=0: 0..4.
    #   j'=1: 5..9.
    #   So for j=2 (j'=0), v_idx 0..4 -> 10..14.
    #   For j=3 (j'=1), v_idx 5..9 -> 15..19.
    # w_idx = k*3+i. Same.
    
    U[2, 28:56] = U_325[0, :]
    U[3, 28:56] = U_325[1, :]
    U[6, 28:56] = U_325[2, :]
    U[7, 28:56] = U_325[3, :]
    U[10, 28:56] = U_325[4, :]
    U[11, 28:56] = U_325[5, :]
    
    V[10, 28:56] = V_325[0, :]
    V[11, 28:56] = V_325[1, :]
    V[12, 28:56] = V_325[2, :]
    V[13, 28:56] = V_325[3, :]
    V[14, 28:56] = V_325[4, :]
    V[15, 28:56] = V_325[5, :]
    V[16, 28:56] = V_325[6, :]
    V[17, 28:56] = V_325[7, :]
    V[18, 28:56] = V_325[8, :]
    V[19, 28:56] = V_325[9, :]
    
    W[:, 28:56] = W_325
    
    # Verify
    if not check_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        base_rank = n * m * p
        U = np.zeros((n * m, base_rank), dtype=int)
        V = np.zeros((m * p, base_rank), dtype=int)
        W = np.zeros((p * n, base_rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    # Return immediately in fallback? No, let's just return the fallback
                    # But we need to return the dict.
                    pass
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
