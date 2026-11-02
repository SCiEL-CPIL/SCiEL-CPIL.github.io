"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    # U shape (n*m, rank) = (6, rank)
    # V shape (m*p, rank) = (21, rank)
    # W shape (p*n, rank) = (14, rank)
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for 2x2 by 2x2
    # a_idx: [A11, A12, A21, A22] global indices in U (row)
    # b_idx: [B11, B12, B21, B22] global indices in V (row)
    # c_idx: [C11, C12, C21, C22] global indices in W (row)
    def add_strassen(a_idx, b_idx, c_idx):
        nonlocal col
        # M1 = (A11 + A22)(B11 + B22)
        U[a_idx[0]][col] += 1
        U[a_idx[2]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[2]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M2 = (A21 + A22)B11
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        W[c_idx[3]][col] -= 1
        col += 1
        
        # M3 = A11(B12 - B22)
        U[a_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[2]][col] -= 1
        W[c_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M4 = A22(B21 - B11)
        U[a_idx[3]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[0]][col] -= 1
        W[c_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M5 = (A11 + A12)B22
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[2]][col] += 1
        W[c_idx[0]][col] -= 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[a_idx[2]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[a_idx[1]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[2]][col] += 1
        V[b_idx[2]][col] += 1 # Wait, B22 is index 2? 
        # Check indices: B11=0, B12=1, B21=2, B22=3
        # M7 uses B21 and B22.
        # V[b_idx[2]] is B21. V[b_idx[3]] is B22.
        # Correcting above:
        # In M7: (A12 - A22)(B21 + B22)
        # U[A12] += 1, U[A22] -= 1
        # V[B21] += 1, V[B22] += 1
        # W[C11] += 1, W[C22] += 1
        
        # Let's re-verify M7 indices in the code above.
        # I wrote: V[b_idx[2]][col] += 1 twice? No, I need to fix the code block below.
        
        pass # Placeholder, will implement correctly in final code

    # Correct Strassen Implementation
    def add_strassen_correct(a_idx, b_idx, c_idx):
        nonlocal col
        # Indices: 0:11, 1:12, 2:21, 3:22
        
        # M1 = (A11 + A22)(B11 + B22)
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M2 = (A21 + A22)B11
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        W[c_idx[3]][col] -= 1
        col += 1
        
        # M3 = A11(B12 - B22)
        U[a_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] -= 1
        W[c_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M4 = A22(B21 - B11)
        U[a_idx[3]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[0]][col] -= 1
        W[c_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M5 = (A11 + A12)B22
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] -= 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[a_idx[2]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[a_idx[1]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1

    # Helper to add 2x2 by 2x1 schoolbook
    # a_idx: [A11, A12, A21, A22]
    # b_idx: [B10, B20]
    # c_idx: [C0, C1]
    def add_schoolbook_2x2x1(a_idx, b_idx, c_idx):
        nonlocal col
        # C0 = A11*B10 + A12*B20
        # C1 = A21*B10 + A22*B20
        
        # M1: A11 * B10 -> C0
        U[a_idx[0]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1
        
        # M2: A12 * B20 -> C0
        U[a_idx[1]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1
        
        # M3: A21 * B10 -> C1
        U[a_idx[2]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M4: A22 * B20 -> C1
        U[a_idx[3]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[1]][col] += 1
        col += 1

    # Helper to add 2x1 by 1x7 outer product
    # a_idx: [A1, A2] (indices in U for col 0 of A2x3? No, A is 2x3. A1 is col 0)
    # b_idx: [B10...B16] (indices in V for row 0 of B3x7)
    # c_idx: [C00...C16] (indices in W)
    def add_outer_product(a_idx, b_idx, c_idx):
        nonlocal col
        # C_ik = A_i * B_k
        # Rank 1 tensor.
        # U_i += 1, V_k += 1, W_ik += 1
        # Wait, if we use rank 1, we add 1 to all relevant U, V, W entries.
        for i in range(2):
            U[a_idx[i]][col] += 1
        for k in range(7):
            V[b_idx[k]][col] += 1
        for i in range(2):
            for k in range(7):
                # W index is k*n + i
                W[k * n + i][col] += 1
        col += 1

    # Global Index Mapping
    # A is 2x3. Indices in U: i*m + j = i*3 + j
    # B is 3x7. Indices in V: j*p + k = j*7 + k
    # C is 2x7. Indices in W: k*n + i = k*2 + i

    # Part 1: A1 (2x1) * B1 (1x7)
    # A1 is A[:, 0] -> U indices 0, 1
    # B1 is B[0, :] -> V indices 0..6
    # C1 is C[:, 0..6] -> W indices k*2 + i
    a1_idx = [0, 1]
    b1_idx = [0, 1, 2, 3, 4, 5, 6]
    c1_idx = [k * 2 + i for i in range(2) for k in range(7)] # Order doesn't matter for W update loop
    add_outer_product(a1_idx, b1_idx, c1_idx)

    # Part 2: A2 (2x2) * B2 (2x7)
    # A2 is A[:, 1:3] -> U indices:
    # A[0,1]=3, A[0,2]=4, A[1,1]=6, A[1,2]=7
    # Order for Strassen: [A11, A12, A21, A22]
    # A11=A[0,1]=3, A12=A[0,2]=4, A21=A[1,1]=6, A22=A[1,2]=7
    a2_idx = [3, 4, 6, 7]
    
    # B2 is B[1:3, :] -> V indices:
    # B[1, k] = 1*7 + k = 7+k
    # B[2, k] = 2*7 + k = 14+k
    
    # Subpart 2.1: B2[:, 0:2]
    # B[1,0]=7, B[1,1]=8, B[2,0]=14, B[2,1]=15
    # Order: [B11, B12, B21, B22]
    b21_idx = [7, 8, 14, 15]
    # C cols 0,1
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    c21_idx = [0, 1, 2, 3]
    add_strassen_correct(a2_idx, b21_idx, c21_idx)

    # Subpart 2.2: B2[:, 2:4]
    # B[1,2]=9, B[1,3]=10, B[2,2]=16, B[2,3]=17
    b22_idx = [9, 10, 16, 17]
    # C cols 2,3
    # C[0,2]=4, C[0,3]=5, C[1,2]=6, C[1,3]=7
    c22_idx = [4, 5, 6, 7]
    add_strassen_correct(a2_idx, b22_idx, c22_idx)

    # Subpart 2.3a: B2[:, 4] (2x1)
    # B[1,4]=11, B[2,4]=18
    b23a_idx = [11, 18]
    # C col 4
    # C[0,4]=8, C[1,4]=9
    c23a_idx = [8, 9]
    add_schoolbook_2x2x1(a2_idx, b23a_idx, c23a_idx)

    # Subpart 2.3b: B2[:, 5:7]
    # B[1,5]=12, B[1,6]=13, B[2,5]=19, B[2,6]=20
    b23b_idx = [12, 13, 19, 20]
    # C cols 5,6
    # C[0,5]=10, C[0,6]=11, C[1,5]=12, C[1,6]=13
    c23b_idx = [10, 11, 12, 13]
    add_strassen_correct(a2_idx, b23b_idx, c23b_idx)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
