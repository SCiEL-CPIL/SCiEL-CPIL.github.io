import time
import sys

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 150.0
    
    # Helper to create zero matrices
    def zero_mat(r, c):
        return [[0.0] * c for _ in range(r)]

    # Base Case 1: Strassen 2x2x2 (Rank 7)
    # A is 2x2 (idx i*2+j), B is 2x2 (idx j*2+k), C is 2x2 (idx k*2+i)
    # U: 4x7, V: 4x7, W: 4x7
    U_s = zero_mat(4, 7)
    V_s = zero_mat(4, 7)
    W_s = zero_mat(4, 7)
    
    # Strassen terms for C = A*B
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)(B00+B01)
    # M6 = (A10-A11)(B10+B11)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Term 0: M1
    U_s[0][0] = 1; U_s[3][0] = 1
    V_s[0][0] = 1; V_s[3][0] = 1
    W_s[0][0] = 1; W_s[3][0] = 1
    
    # Term 1: M2
    U_s[2][1] = 1; U_s[3][1] = 1
    V_s[0][1] = 1
    W_s[2][1] = 1; W_s[3][1] = -1
    
    # Term 2: M3
    U_s[0][2] = 1
    V_s[1][2] = 1; V_s[3][2] = -1
    W_s[1][2] = 1; W_s[3][2] = 1
    
    # Term 3: M4
    U_s[3][3] = 1
    V_s[2][3] = 1; V_s[0][3] = -1
    W_s[0][3] = 1; W_s[2][3] = 1
    
    # Term 4: M5
    U_s[0][4] = 1; U_s[2][4] = 1
    V_s[0][4] = 1; V_s[1][4] = 1
    W_s[0][4] = -1; W_s[1][4] = 1
    
    # Term 5: M6
    U_s[2][5] = 1; U_s[3][5] = -1
    V_s[2][5] = 1; V_s[3][5] = 1
    W_s[2][5] = 1; W_s[3][5] = 1
    
    # Term 6: M7
    U_s[1][6] = 1; U_s[3][6] = -1
    V_s[1][6] = 1; V_s[3][6] = 1
    W_s[0][6] = 1; W_s[1][6] = 1

    # Base Case 2: Schoolbook 2x2x1 (Rank 4)
    # A is 2x2 (idx i*2+j), B is 2x1 (idx j*1+k), C is 2x1 (idx k*2+i)
    # U: 4x4, V: 2x4, W: 2x4
    U_k = zero_mat(4, 4)
    V_k = zero_mat(2, 4)
    W_k = zero_mat(2, 4)
    
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    
    # Term 0: A00*B0
    U_k[0][0] = 1
    V_k[0][0] = 1
    W_k[0][0] = 1
    
    # Term 1: A01*B1
    U_k[1][1] = 1
    V_k[1][1] = 1
    W_k[0][1] = 1
    
    # Term 2: A10*B0
    U_k[2][2] = 1
    V_k[0][2] = 1
    W_k[1][2] = 1
    
    # Term 3: A11*B1
    U_k[3][3] = 1
    V_k[1][3] = 1
    W_k[1][3] = 1

    # Combine 2x2x2 (Strassen) and 2x2x1 (Schoolbook) to form 2x2x3
    # A is 2x2, B is 2x3, C is 2x3
    # Rank = 7 + 4 = 11
    # U: 4x11, V: 6x11, W: 6x11
    
    U_223 = zero_mat(4, 11)
    V_223 = zero_mat(6, 11)
    W_223 = zero_mat(6, 11)
    
    # Map Strassen (cols 0,1 of B and C)
    # Strassen V (2x2) indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # 2x3 V indices: 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    v_map_s_22_to_23 = [0, 1, 3, 4]
    
    # Strassen W (2x2) indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
    # 2x3 W indices: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1), 4:(0,2), 5:(1,2)
    w_map_s_22_to_23 = [0, 1, 2, 3]
    
    for r in range(7):
        for i in range(4):
            U_223[i][r] = U_s[i][r]
            V_223[v_map_s_22_to_23[i]][r] = V_s[i][r]
            W_223[w_map_s_22_to_23[i]][r] = W_s[i][r]
            
    # Map Schoolbook (col 2 of B and C)
    # Schoolbook V (2x1) indices: 0:(0,0), 1:(1,0)
    # 2x3 V indices for col 2: 2:(0,2), 5:(1,2)
    v_map_k_21_to_23 = [2, 5]
    
    # Schoolbook W (2x1) indices: 0:(0,0), 1:(1,0)
    # 2x3 W indices for col 2: 4:(0,2), 5:(1,2)
    w_map_k_21_to_23 = [4, 5]
    
    for r in range(4):
        for i in range(4):
            U_223[i][7+r] = U_k[i][r]
            
    for r in range(4):
        for i in range(2):
            V_223[v_map_k_21_to_23[i]][7+r] = V_k[i][r]
            W_223[w_map_k_21_to_23[i]][7+r] = W_k[i][r]

    # Combine two 2x2x3 blocks to form 2x2x6
    # A is 2x2, B is 2x6, C is 2x6
    # Rank = 11 + 11 = 22
    # U: 4x22, V: 12x22, W: 12x22
    
    U_226 = zero_mat(4, 22)
    V_226 = zero_mat(12, 22)
    W_226 = zero_mat(12, 22)
    
    # Block 1: Cols 0-2 of B and C
    # 2x3 V indices: 0-5
    # 2x6 V indices: 0-5 (cols 0-2), 6-11 (cols 3-5)
    v_map_23_to_26_1 = [0, 1, 2, 3, 4, 5]
    
    # 2x3 W indices: 0-5
    # 2x6 W indices: 0-5 (cols 0-2), 6-11 (cols 3-5)
    w_map_23_to_26_1 = [0, 1, 2, 3, 4, 5]
    
    for r in range(11):
        for i in range(4):
            U_226[i][r] = U_223[i][r]
        for i in range(6):
            V_226[v_map_23_to_26_1[i]][r] = V_223[i][r]
            W_226[w_map_23_to_26_1[i]][r] = W_223[i][r]
            
    # Block 2: Cols 3-5 of B and C
    # 2x3 V indices: 0-5
    # 2x6 V indices: 6-11
    v_map_23_to_26_2 = [6, 7, 8, 9, 10, 11]
    
    # 2x3 W indices: 0-5
    # 2x6 W indices: 6-11
    w_map_23_to_26_2 = [6, 7, 8, 9, 10, 11]
    
    for r in range(11):
        for i in range(4):
            U_226[i][11+r] = U_223[i][r]
        for i in range(6):
            V_226[v_map_23_to_26_2[i]][11+r] = V_223[i][r]
            W_226[w_map_23_to_26_2[i]][11+r] = W_223[i][r]

    # Combine two 2x2x6 blocks to form 2x4x6
    # A is 2x4, B is 4x6, C is 2x6
    # Rank = 22 + 22 = 44
    # U: 8x44, V: 24x44, W: 12x44
    
    U_final = zero_mat(8, 44)
    V_final = zero_mat(24, 44)
    W_final = zero_mat(12, 44)
    
    # Block 1: Rows 0-1 of A, Rows 0-1 of B
    # 2x2x6 U indices: 0-3
    # 2x4x6 U indices: 0-3 (rows 0-1), 4-7 (rows 2-3)
    # Note: A is 2x4. U index is i*4+j.
    # For rows 0-1 of A (i=0,1), j=0,1 map to U indices 0,1,4,5?
    # Wait. A is 2x4.
    # U dimension is n*m = 2*4 = 8.
    # Index in U is i*m + j = i*4 + j.
    # Strassen/Schoolbook U was for A 2x2. Index i*2+j.
    # So U_s[0] (i=0,j=0) -> U_final[0*4+0] = 0
    # U_s[1] (i=0,j=1) -> U_final[0*4+1] = 1
    # U_s[2] (i=1,j=0) -> U_final[1*4+0] = 4
    # U_s[3] (i=1,j=1) -> U_final[1*4+1] = 5
    
    u_map_22_to_24_1 = [0, 1, 4, 5]
    
    # 2x2x6 V indices: 0-11 (B is 2x6)
    # 4x6 V indices: 0-11 (rows 0-1), 12-23 (rows 2-3)
    # B is 4x6. Index j*6+k.
    # Rows 0-1 of B: j=0,1. Indices 0-5, 6-11.
    v_map_26_to_46_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    # 2x2x6 W indices: 0-11 (C is 2x6)
    # 2x4x6 W indices: 0-11 (C is 2x6)
    # C is 2x6. Index k*2+i.
    # The C matrix is the same for both blocks of A?
    # Yes, C = A*B. If we split A into A1 (rows 0-1) and A2 (rows 2-3),
    # C = A1*B1 + A2*B2.
    # The output C is 2x6.
    # The W matrix for the decomposition of C = A1*B1 uses the same W indices as C = A2*B2?
    # Yes, the output C is the same.
    w_map_26_to_26_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    for r in range(22):
        for i in range(4):
            U_final[u_map_22_to_24_1[i]][r] = U_226[i][r]
        for i in range(12):
            V_final[v_map_26_to_46_1[i]][r] = V_226[i][r]
            W_final[w_map_26_to_26_1[i]][r] = W_226[i][r]
            
    # Block 2: Rows 2-3 of A, Rows 2-3 of B
    # U indices for rows 2-3 of A (i=2,3 in 4x4 view? No, A is 2x4)
    # Wait, A is 2x4.
    # My decomposition splits A into two 2x2 blocks?
    # No, I split A (2x4) into A_top (2x2, cols 0-1) and A_bot (2x2, cols 2-3)?
    # Or A_left (2x2, rows 0-1) and A_right (2x2, rows 2-3)?
    # A is 2 rows, 4 cols.
    # The multiplication is (2x4) * (4x6) = (2x6).
    # I can split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    # And B into B1 (2x6, rows 0-1) and B2 (2x6, rows 2-3).
    # C = A1*B1 + A2*B2.
    
    # So for Block 2:
    # A2 is 2x2.
    # U indices for A2 (cols 2-3 of A):
    # i=0, j=2 -> 0*4+2 = 2
    # i=0, j=3 -> 0*4+3 = 3
    # i=1, j=2 -> 1*4+2 = 6
    # i=1, j=3 -> 1*4+3 = 7
    u_map_22_to_24_2 = [2, 3, 6, 7]
    
    # B2 is 2x6 (rows 2-3 of B).
    # V indices for B2 (rows 2-3 of B):
    # j=2, k=0..5 -> 2*6+0..5 = 12..17
    # j=3, k=0..5 -> 3*6+0..5 = 18..23
    v_map_26_to_46_2 = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    
    # W indices are the same as Block 1 because C is the same.
    w_map_26_to_26_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    for r in range(22):
        for i in range(4):
            U_final[u_map_22_to_24_2[i]][22+r] = U_226[i][r]
        for i in range(12):
            V_final[v_map_26_to_46_2[i]][22+r] = V_226[i][r]
            W_final[w_map_26_to_26_2[i]][22+r] = W_226[i][r]
            
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
