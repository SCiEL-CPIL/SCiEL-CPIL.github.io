import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We have 4 blocks of 2x2 multiplication for the 2x4 x 4x4 part.
    # Block 1: A_L * B_LL -> C_LL (cols 0,1 of C1)
    # Block 2: A_L * B_LR -> C_LR (cols 2,3 of C1)
    # Block 3: A_R * B_RL -> C_RL (cols 0,1 of C2? No)
    
    # Let's re-verify the block structure for 2x4 x 4x4.
    # A is 2x4. B is 4x4.
    # A = [A_L, A_R] where A_L, A_R are 2x2.
    # B = [[B_LL, B_LR], [B_RL, B_RR]] where blocks are 2x2.
    # C = A B = [A_L B_LL + A_R B_RL, A_L B_LR + A_R B_RR]
    # Let P1 = A_L B_LL, P2 = A_R B_RL, P3 = A_L B_LR, P4 = A_R B_RR.
    # C_left = P1 + P2
    # C_right = P3 + P4
    
    # For the full 2x4 x 4x8 problem:
    # B is 4x8. Split into B1 (4x4) and B2 (4x4).
    # C = [C1, C2] where C1 = A B1, C2 = A B2.
    # C1 uses P1, P2, P3, P4 from B1.
    # C2 uses P5, P6, P7, P8 from B2.
    
    # Total 8 Strassen products. Each rank 7. Total 56.
    
    # Mapping of indices
    # A indices: i in 0..1, j in 0..3. idx = i*4 + j
    # B indices: j in 0..3, k in 0..7. idx = j*8 + k
    # W indices: k in 0..7, i in 0..1. idx = k*2 + i
    
    # Define the 8 products
    # Product 1: A_L * B1_LL
    # A_L is A[:, 0:2]
    # B1_LL is B[0:2, 0:2]
    
    # We will iterate over the 8 products and 7 Strassen terms.
    
    products = []
    # Product 1: A_L (cols 0,1) * B1_LL (rows 0,1; cols 0,1) -> C cols 0,1
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1]
    # B1_LL: B[0,0], B[0,1], B[1,0], B[1,1]
    # Output C[0,0], C[0,1], C[1,0], C[1,1]
    
    # Product 2: A_R (cols 2,3) * B1_RL (rows 2,3; cols 0,1) -> C cols 0,1
    # A_R: A[0,2], A[0,3], A[1,2], A[1,3]
    # B1_RL: B[2,0], B[2,1], B[3,0], B[3,1]
    # Output C[0,0], C[0,1], C[1,0], C[1,1] (Added to P1)
    
    # Product 3: A_L (cols 0,1) * B1_LR (rows 0,1; cols 2,3) -> C cols 2,3
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1]
    # B1_LR: B[0,2], B[0,3], B[1,2], B[1,3]
    # Output C[0,2], C[0,3], C[1,2], C[1,3]
    
    # Product 4: A_R (cols 2,3) * B1_RR (rows 2,3; cols 2,3) -> C cols 2,3
    # A_R: A[0,2], A[0,3], A[1,2], A[1,3]
    # B1_RR: B[2,2], B[2,3], B[3,2], B[3,3]
    # Output C[0,2], C[0,3], C[1,2], C[1,3] (Added to P3)
    
    # Product 5: A_L * B2_LL -> C cols 4,5
    # Product 6: A_R * B2_RL -> C cols 4,5
    # Product 7: A_L * B2_LR -> C cols 6,7
    # Product 8: A_R * B2_RR -> C cols 6,7
    
    # Helper to add a Strassen term
    def add_strassen_term(prod_idx, term_idx, a_indices, b_indices, w_indices, coeffs):
        # a_indices: list of (U_idx, coeff)
        # b_indices: list of (V_idx, coeff)
        # w_indices: list of (W_idx, coeff)
        col = prod_idx * 7 + term_idx
        for u_idx, c in a_indices:
            U[u_idx][col] += c
        for v_idx, c in b_indices:
            V[v_idx][col] += c
        for w_idx, c in w_indices:
            W[w_idx][col] += c
            
    # Strassen definitions
    # M1: A11+A22, B11+B22 -> C11+1, C22+1
    # M2: A21+A22, B11 -> C21+1, C22-1
    # M3: A11, B12-B22 -> C12+1, C22+1
    # M4: A22, B21-B11 -> C11+1, C21+1
    # M5: A11+A12, B22 -> C11-1, C12+1
    # M6: A21-A11, B11+B12 -> C22+1
    # M7: A12-A22, B21+B22 -> C11+1
    
    # We need to map A11, A12, A21, A22 to U indices
    # and B11, B12, B21, B22 to V indices
    # and C11, C12, C21, C22 to W indices
    
    # For each product, we define the mapping.
    
    # Product 1: A_L * B1_LL
    # A_L: A11=A[0,0]=0, A12=A[0,1]=1, A21=A[1,0]=4, A22=A[1,1]=5
    # B1_LL: B11=B[0,0]=0, B12=B[0,1]=1, B21=B[1,0]=8, B22=B[1,1]=9
    # C: C11=C[0,0]=0, C12=C[0,1]=2, C21=C[1,0]=1, C22=C[1,1]=3
    
    # Product 2: A_R * B1_RL
    # A_R: A11=A[0,2]=2, A12=A[0,3]=3, A21=A[1,2]=6, A22=A[1,3]=7
    # B1_RL: B11=B[2,0]=16, B12=B[2,1]=17, B21=B[3,0]=24, B22=B[3,1]=25
    # C: C11=C[0,0]=0, C12=C[0,1]=2, C21=C[1,0]=1, C22=C[1,1]=3
    
    # Product 3: A_L * B1_LR
    # A_L: A11=0, A12=1, A21=4, A22=5
    # B1_LR: B11=B[0,2]=2, B12=B[0,3]=3, B21=B[1,2]=10, B22=B[1,3]=11
    # C: C11=C[0,2]=4, C12=C[0,3]=6, C21=C[1,2]=5, C22=C[1,3]=7
    
    # Product 4: A_R * B1_RR
    # A_R: A11=2, A12=3, A21=6, A22=7
    # B1_RR: B11=B[2,2]=18, B12=B[2,3]=19, B21=B[3,2]=26, B22=B[3,3]=27
    # C: C11=C[0,2]=4, C12=C[0,3]=6, C21=C[1,2]=5, C22=C[1,3]=7
    
    # Product 5: A_L * B2_LL (B2 is cols 4-7)
    # A_L: A11=0, A12=1, A21=4, A22=5
    # B2_LL: B11=B[0,4]=4, B12=B[0,5]=5, B21=B[1,4]=12, B22=B[1,5]=13
    # C: C11=C[0,4]=8, C12=C[0,5]=10, C21=C[1,4]=9, C22=C[1,5]=11
    
    # Product 6: A_R * B2_RL
    # A_R: A11=2, A12=3, A21=6, A22=7
    # B2_RL: B11=B[2,4]=20, B12=B[2,5]=21, B21=B[3,4]=28, B22=B[3,5]=29
    # C: C11=C[0,4]=8, C12=C[0,5]=10, C21=C[1,4]=9, C22=C[1,5]=11
    
    # Product 7: A_L * B2_LR
    # A_L: A11=0, A12=1, A21=4, A22=5
    # B2_LR: B11=B[0,6]=6, B12=B[0,7]=7, B21=B[1,6]=14, B22=B[1,7]=15
    # C: C11=C[0,6]=12, C12=C[0,7]=14, C21=C[1,6]=13, C22=C[1,7]=15
    
    # Product 8: A_R * B2_RR
    # A_R: A11=2, A12=3, A21=6, A22=7
    # B2_RR: B11=B[2,6]=22, B12=B[2,7]=23, B21=B[3,6]=30, B22=B[3,7]=31
    # C: C11=C[0,6]=12, C12=C[0,7]=14, C21=C[1,6]=13, C22=C[1,7]=15
    
    mappings = [
        # (A11, A12, A21, A22, B11, B12, B21, B22, C11, C12, C21, C22)
        (0, 1, 4, 5, 0, 1, 8, 9, 0, 2, 1, 3),   # P1
        (2, 3, 6, 7, 16, 17, 24, 25, 0, 2, 1, 3), # P2
        (0, 1, 4, 5, 2, 3, 10, 11, 4, 6, 5, 7),   # P3
        (2, 3, 6, 7, 18, 19, 26, 27, 4, 6, 5, 7), # P4
        (0, 1, 4, 5, 4, 5, 12, 13, 8, 10, 9, 11), # P5
        (2, 3, 6, 7, 20, 21, 28, 29, 8, 10, 9, 11), # P6
        (0, 1, 4, 5, 6, 7, 14, 15, 12, 14, 13, 15), # P7
        (2, 3, 6, 7, 22, 23, 30, 31, 12, 14, 13, 15)  # P8
    ]
    
    for p_idx, m in enumerate(mappings):
        a11, a12, a21, a22 = m[0], m[1], m[2], m[3]
        b11, b12, b21, b22 = m[4], m[5], m[6], m[7]
        c11, c12, c21, c22 = m[8], m[9], m[10], m[11]
        
        # Term 0: M1
        add_strassen_term(p_idx, 0, [(a11, 1), (a22, 1)], [(b11, 1), (b22, 1)], [(c11, 1), (c22, 1)])
        # Term 1: M2
        add_strassen_term(p_idx, 1, [(a21, 1), (a22, 1)], [(b11, 1)], [(c21, 1), (c22, -1)])
        # Term 2: M3
        add_strassen_term(p_idx, 2, [(a11, 1)], [(b12, 1), (b22, -1)], [(c12, 1), (c22, 1)])
        # Term 3: M4
        add_strassen_term(p_idx, 3, [(a22, 1)], [(b21, 1), (b11, -1)], [(c11, 1), (c21, 1)])
        # Term 4: M5
        add_strassen_term(p_idx, 4, [(a11, 1), (a12, 1)], [(b22, 1)], [(c11, -1), (c12, 1)])
        # Term 5: M6
        add_strassen_term(p_idx, 5, [(a21, 1), (a11, -1)], [(b11, 1), (b12, 1)], [(c22, 1)])
        # Term 6: M7
        add_strassen_term(p_idx, 6, [(a12, 1), (a22, -1)], [(b21, 1), (b22, 1)], [(c11, 1)])
        
    return {"U": U, "V": V, "W": W}
