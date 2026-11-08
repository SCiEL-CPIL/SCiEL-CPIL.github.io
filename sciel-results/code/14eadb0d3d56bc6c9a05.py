import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 49 for 4x4x4
    # Rank 28 for 4x4x2
    total_rank = 49 + 28
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a local decomposition
    def add_decomp(U_local, V_local, W_local, offset, a_map, b_map, c_map):
        # a_map, b_map, c_map: dict from local index to global index
        # U_local[i][k] is coefficient for local A_i in rank k
        for i_local in range(len(U_local)):
            i_global = a_map[i_local]
            for k in range(len(U_local[0])):
                U[i_global][offset + k] += U_local[i_local][k]
        for j_local in range(len(V_local)):
            j_global = b_map[j_local]
            for k in range(len(V_local[0])):
                V[j_global][offset + k] += V_local[j_local][k]
        for l_local in range(len(W_local)):
            l_global = c_map[l_local]
            for k in range(len(W_local[0])):
                W[l_global][offset + k] += W_local[l_local][k]

    # Strassen 2x2x2 vectors
    # U: 4x7, V: 4x7, W: 4x7
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Part 1: 4x4x4 (Rank 49)
    # 7 Strassen steps
    # Each step is a 2x2x2 multiplication of P and Q
    # P and Q are 2x2 matrices formed by linear combos of A and B blocks
    
    # Define linear combos for P and Q for each step s
    # P_s[i,j] = sum c * A_block
    # We map local 2x2 indices (0..3) to global 4x4 indices (0..15)
    
    # Step 1: M1 = (A00+A11)(B00+B11)
    # P1 = A00+A11, Q1 = B00+B11
    # A00 is 2x2 at (0,0), A11 is 2x2 at (2,2)
    # P1[0,0] = A[0,0] + A[2,2]
    # P1[0,1] = A[0,1] + A[2,3]
    # P1[1,0] = A[1,0] + A[3,2]
    # P1[1,1] = A[1,1] + A[3,3]
    
    # We need to construct the U vectors for P1.
    # U_local for 2x2x2 has 4 rows.
    # Row 0 corresponds to P[0,0].
    # So U_global[0,0] (for A[0,0]) gets U_local[0, k].
    # U_global[2,2] (for A[2,2]) gets U_local[0, k].
    
    # Let's define the mappings for A, B, C for each of the 7 steps.
    
    # Step 1: (A00+A11)(B00+B11)
    # A map: local 0->A00, local 1->A01, local 2->A10, local 3->A11? No.
    # P is 2x2. Local index 0 is P[0,0], 1 is P[0,1], 2 is P[1,0], 3 is P[1,1].
    # P[0,0] = A[0,0] + A[2,2]
    # P[0,1] = A[0,1] + A[2,3]
    # P[1,0] = A[1,0] + A[3,2]
    # P[1,1] = A[1,1] + A[3,3]
    
    # We can't just use a simple map because each local entry is a sum.
    # We need to add the contributions.
    
    def add_strassen_block(P_coeffs, Q_coeffs, C_coeffs, offset):
        # P_coeffs: list of 4 lists, each list is (global_idx, coeff)
        # Q_coeffs: list of 4 lists
        # C_coeffs: list of 4 lists
        for k in range(7):
            for i_local in range(4):
                val = S_U[i_local][k]
                if val != 0:
                    for idx, c in P_coeffs[i_local]:
                        U[idx][offset + k] += val * c
            for j_local in range(4):
                val = S_V[j_local][k]
                if val != 0:
                    for idx, c in Q_coeffs[j_local]:
                        V[idx][offset + k] += val * c
            for l_local in range(4):
                val = S_W[l_local][k]
                if val != 0:
                    for idx, c in C_coeffs[l_local]:
                        W[idx][offset + k] += val * c

    # Define coefficients for the 7 steps
    # A indices: 0..15 (row * 4 + col)
    # B indices: 0..23 (row * 6 + col) -- Wait, B is 4x6.
    # For 4x4x4 part, B is 4x4. Indices 0..15 (row * 4 + col).
    # C indices: 0..15 (row * 4 + col).
    
    # Step 1: M1 = (A00+A11)(B00+B11)
    P1 = [
        [(0, 1), (8, 1)],   # P[0,0] = A[0,0] + A[2,2] (A[2,2] is idx 2*4+2=10? No. A is 4x4. A[i,j] idx i*4+j. A[2,2] is 10. A[0,0] is 0.)
        # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5.
        # A[2,2]=10, A[2,3]=11, A[3,2]=14, A[3,3]=15.
        [(0, 1), (10, 1)],
        [(1, 1), (11, 1)],  # P[0,1] = A[0,1] + A[2,3]
        [(4, 1), (14, 1)],  # P[1,0] = A[1,0] + A[3,2]
        [(5, 1), (15, 1)]   # P[1,1] = A[1,1] + A[3,3]
    ]
    Q1 = [
        [(0, 1), (10, 1)],  # Q[0,0] = B[0,0] + B[2,2]
        [(1, 1), (11, 1)],
        [(4, 1), (14, 1)],
        [(5, 1), (15, 1)]
    ]
    # C1 = M1 contributes to C00, C01, C10, C11 with coeff 1?
    # C00 = M1 + ...
    # So C[0,0] gets M[0,0] with coeff 1.
    # C[0,1] gets M[0,1] with coeff 1?
    # M is 2x2. M[0,0] is C[0,0] block?
    # No, M is the result of P*Q.
    # P is 2x2, Q is 2x2. M is 2x2.
    # M[0,0] corresponds to C[0,0] (top-left 2x2 block).
    # So C[0,0] (idx 0) gets M[0,0] (local 0) with coeff 1.
    # C[0,1] (idx 1) gets M[0,1] (local 1) with coeff 1.
    # C[1,0] (idx 4) gets M[1,0] (local 2) with coeff 1.
    # C[1,1] (idx 5) gets M[1,1] (local 3) with coeff 1.
    C1 = [
        [(0, 1)],
        [(1, 1)],
        [(4, 1)],
        [(5, 1)]
    ]
    add_strassen_block(P1, Q1, C1, 0)
    
    # Step 2: M2 = (A10+A11)B00
    # P2 = A10+A11
    P2 = [
        [(4, 1), (8, 1)],   # A[1,0]+A[2,2]? No. A10 is A[2,0:2,2]?
        # A10 is bottom-left 2x2. Rows 2,3. Cols 0,1.
        # A[2,0]=8, A[2,1]=9, A[3,0]=12, A[3,1]=13.
        # A11 is bottom-right. A[2,2]=10, A[2,3]=11, A[3,2]=14, A[3,3]=15.
        # P2[0,0] = A[2,0] + A[2,2] = 8 + 10
        [(8, 1), (10, 1)],
        [(9, 1), (11, 1)],  # P2[0,1] = A[2,1] + A[2,3]
        [(12, 1), (14, 1)], # P2[1,0] = A[3,0] + A[3,2]
        [(13, 1), (15, 1)]  # P2[1,1] = A[3,1] + A[3,3]
    ]
    # Q2 = B00
    Q2 = [
        [(0, 1)],
        [(1, 1)],
        [(4, 1)],
        [(5, 1)]
    ]
    # C2 = M2 contributes to C10, C11 with coeff -1?
    # C10 = M2 + M3. So C[1,0] (idx 4) gets M[1,0] (local 2) with coeff 1.
    # C11 = M1 - M2 + ... So C[1,1] (idx 5) gets M[1,1] (local 3) with coeff -1.
    C2 = [
        [],
        [],
        [(4, 1)],
        [(5, -1)]
    ]
    add_strassen_block(P2, Q2, C2, 7)
    
    # Step 3: M3 = A00(B10-B11)
    P3 = [
        [(0, 1)],
        [(1, 1)],
        [(4, 1)],
        [(5, 1)]
    ]
    # Q3 = B10-B11
    # B10 is bottom-left of B (4x4). B[2,0:2,2].
    # B11 is bottom-right.
    Q3 = [
        [(8, 1), (10, -1)],
        [(9, 1), (11, -1)],
        [(12, 1), (14, -1)],
        [(13, 1), (15, -1)]
    ]
    # C3 = M3 contributes to C10, C11 with coeff 1.
    C3 = [
        [],
        [],
        [(4, 1)],
        [(5, 1)]
    ]
    add_strassen_block(P3, Q3, C3, 14)
    
    # Step 4: M4 = A11(B01-B00)
    P4 = [
        [(10, 1)],
        [(11, 1)],
        [(14, 1)],
        [(15, 1)]
    ]
    # Q4 = B01-B00
    Q4 = [
        [(4, -1), (0, 1)],  # B[1,0]-B[0,0]? No. B01 is top-right. B[0,2:0,4].
        # B00 is top-left. B[0,0:0,2].
        # Q4[0,0] = B[0,2] - B[0,0]
        [(2, 1), (0, -1)],
        [(3, 1), (1, -1)],
        [(6, 1), (4, -1)],
        [(7, 1), (5, -1)]
    ]
    # C4 = M4 contributes to C00, C11 with coeff 1.
    C4 = [
        [(0, 1)],
        [],
        [],
        [(5, 1)]
    ]
    add_strassen_block(P4, Q4, C4, 21)
    
    # Step 5: M5 = (A00+A01)B11
    P5 = [
        [(0, 1), (2, 1)],  # A[0,0]+A[0,2]
        [(1, 1), (3, 1)],
        [(4, 1), (6, 1)],
        [(5, 1), (7, 1)]
    ]
    Q5 = [
        [(10, 1)],
        [(11, 1)],
        [(14, 1)],
        [(15, 1)]
    ]
    # C5 = M5 contributes to C00, C01 with coeff -1, 1.
    C5 = [
        [(0, -1)],
        [(1, 1)],
        [],
        []
    ]
    add_strassen_block(P5, Q5, C5, 28)
    
    # Step 6: M6 = (A10-A00)(B00+B01)
    P6 = [
        [(8, 1), (0, -1)],
        [(9, 1), (1, -1)],
        [(12, 1), (4, -1)],
        [(13, 1), (5, -1)]
    ]
    Q6 = [
        [(0, 1), (2, 1)],
        [(1, 1), (3, 1)],
        [(4, 1), (6, 1)],
        [(5, 1), (7, 1)]
    ]
    # C6 = M6 contributes to C01, C11 with coeff 1.
    C6 = [
        [],
        [(1, 1)],
        [],
        [(5, 1)]
    ]
    add_strassen_block(P6, Q6, C6, 35)
    
    # Step 7: M7 = (A01-A11)(B10+B11)
    P7 = [
        [(2, 1), (10, -1)],
        [(3, 1), (11, -1)],
        [(6, 1), (14, -1)],
        [(7, 1), (15, -1)]
    ]
    Q7 = [
        [(8, 1), (10, 1)],
        [(9, 1), (11, 1)],
        [(12, 1), (14, 1)],
        [(13, 1), (15, 1)]
    ]
    # C7 = M7 contributes to C00, C01 with coeff 1.
    C7 = [
        [(0, 1)],
        [(1, 1)],
        [],
        []
    ]
    add_strassen_block(P7, Q7, C7, 42)
    
    # Part 2: 4x4x2 (Rank 28)
    # Offset 49
    # 4 independent 2x2x2 multiplications
    
    # 1. A00 B0 -> C0 (top-left of C0? No, C0 is top 2 rows of C, which is 2x2 block? No, C is 4x2. C0 is top 2x2 block.)
    # A00 is 2x2. B0 is top 2x2 of B (4x2). B0 is B[0:2, 0:2].
    # C0 is C[0:2, 0:2].
    P_2_1 = [
        [(0, 1)],
        [(1, 1)],
        [(4, 1)],
        [(5, 1)]
    ]
    Q_2_1 = [
        [(0, 1)],
        [(1, 1)],
        [(4, 1)],  # B[2,0]? No. B is 4x2. B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3.
        # B0 is B[0:2, 0:2].
        # B[0,0]=0, B[0,1]=1.
        # B[1,0]=2, B[1,1]=3.
        [(2, 1)],
        [(3, 1)]
    ]
    C_2_1 = [
        [(0, 1)],  # C[0,0]
        [(1, 1)],  # C[0,1]
        [(2, 1)],  # C[1,0]
        [(3, 1)]   # C[1,1]
    ]
    add_strassen_block(P_2_1, Q_2_1, C_2_1, 49)
    
    # 2. A01 B1 -> C0
    # A01 is top-right 2x2. A[0,2:0,4].
    # B1 is bottom 2x2 of B. B[2:4, 0:2].
    # C0 is top 2x2 of C.
    P_2_2 = [
        [(2, 1)],
        [(3, 1)],
        [(6, 1)],
        [(7, 1)]
    ]
    Q_2_2 = [
        [(4, 1)],  # B[2,0]
        [(5, 1)],  # B[2,1]
        [(6, 1)],  # B[3,0]
        [(7, 1)]   # B[3,1]
    ]
    C_2_2 = [
        [(0, 1)],
        [(1, 1)],
        [(2, 1)],
        [(3, 1)]
    ]
    add_strassen_block(P_2_2, Q_2_2, C_2_2, 56)
    
    # 3. A10 B0 -> C1
    # A10 is bottom-left 2x2.
    # B0 is top 2x2 of B.
    # C1 is bottom 2x2 of C. C[2:4, 0:2].
    P_2_3 = [
        [(8, 1)],
        [(9, 1)],
        [(12, 1)],
        [(13, 1)]
    ]
    Q_2_3 = [
        [(0, 1)],
        [(1, 1)],
        [(2, 1)],
        [(3, 1)]
    ]
    C_2_3 = [
        [(8, 1)],  # C[2,0]
        [(9, 1)],  # C[2,1]
        [(10, 1)], # C[3,0]
        [(11, 1)]  # C[3,1]
    ]
    add_strassen_block(P_2_3, Q_2_3, C_2_3, 63)
    
    # 4. A11 B1 -> C1
    P_2_4 = [
        [(10, 1)],
        [(11, 1)],
        [(14, 1)],
        [(15, 1)]
    ]
    Q_2_4 = [
        [(4, 1)],
        [(5, 1)],
        [(6, 1)],
        [(7, 1)]
    ]
    C_2_4 = [
        [(8, 1)],
        [(9, 1)],
        [(10, 1)],
        [(11, 1)]
    ]
    add_strassen_block(P_2_4, Q_2_4, C_2_4, 70)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
