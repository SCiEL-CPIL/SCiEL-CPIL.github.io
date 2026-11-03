"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # A is 2x4, split into A_L (2x2) and A_R (2x2)
    # B is 4x5, split into B_T (2x5) and B_B (2x5)
    # C = A_L B_T + A_R B_B
    # Rank of 2x2x5 is approx 18 (using 2+2+1 split: 7+7+4)
    # Total rank = 18 + 18 = 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)] # 8 x 36
    V = [[0] * rank for _ in range(m * p)] # 20 x 36
    W = [[0] * rank for _ in range(p * n)] # 10 x 36
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # A indices: i in 0,1; j in 0,1 (local)
    # B indices: j in 0,1 (local); k in 0,1 (local)
    # C indices: k in 0,1 (local); i in 0,1 (local)
    def add_strassen_222(a_base_i, a_base_j, b_base_j, b_base_k, w_base_k, w_base_i):
        nonlocal col
        # Strassen coefficients
        # M1 = (a00+a10)(b00+b01)
        # M2 = (a01+a11)(b00+b01)
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a11)(b00+b11)
        # M6 = (a01+a11)(b10-b11)
        # M7 = (a00-a01)(b00+b10)
        
        # Note: My indices for A are i (row), j (col).
        # A is 2x2.
        # a00 is A[0,0], a10 is A[1,0]? No, standard matrix A[i,j].
        # Strassen:
        # M1 = (A00+A10) * (B00+B10) ? No.
        # Standard Strassen for C = A B:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B10)
        # M7 = (A01 - A11)(B10 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Mapping to my tensor indices:
        # U index: i * m + j. Local i, j.
        # V index: j * p + k. Local j, k.
        # W index: k * n + i. Local k, i.
        
        # A entries:
        # A00 -> U[a_base_i + 0, col]
        # A10 -> U[a_base_i + 1, col] (Row 1, Col 0)
        # A01 -> U[a_base_i + 0, col] (Row 0, Col 1) -- Wait, U is indexed by i*m+j.
        # A is 2x2.
        # A[0,0] idx 0. A[0,1] idx 1. A[1,0] idx 2. A[1,1] idx 3.
        # In my U matrix, row is i*m+j.
        # So for A_L (rows 0,1 of A; cols 0,1 of A):
        # A[0,0] is U[0, col]
        # A[0,1] is U[1, col]
        # A[1,0] is U[2, col]
        # A[1,1] is U[3, col]
        
        # B entries:
        # B is 2x2.
        # B[0,0] idx 0. B[0,1] idx 1. B[1,0] idx 2. B[1,1] idx 3.
        # V index j*p+k.
        # For B_T (rows 0,1 of B; cols 0,1 of B):
        # B[0,0] is V[b_base_j + 0, col]
        # B[0,1] is V[b_base_j + 1, col]
        # B[1,0] is V[b_base_j + 2, col]
        # B[1,1] is V[b_base_j + 3, col]
        
        # C entries:
        # C is 2x2.
        # C[0,0] idx 0. C[0,1] idx 1. C[1,0] idx 2. C[1,1] idx 3.
        # W index k*n+i.
        # For C_T (rows 0,1 of C; cols 0,1 of C):
        # C[0,0] is W[w_base_k + 0, col]
        # C[0,1] is W[w_base_k + 1, col]
        # C[1,0] is W[w_base_k + 2, col]
        # C[1,1] is W[w_base_k + 3, col]
        
        # Strassen Steps
        # 1. M1 = (A00+A11)(B00+B11)
        U[a_base_i + 0][col] = 1
        U[a_base_i + 3][col] = 1
        V[b_base_j + 0][col] = 1
        V[b_base_j + 3][col] = 1
        W[w_base_k + 0][col] = 1
        W[w_base_k + 3][col] = 1
        W[w_base_k + 1][col] = 1
        W[w_base_k + 2][col] = 1
        col += 1
        
        # 2. M2 = (A10+A11)B00
        U[a_base_i + 2][col] = 1
        U[a_base_i + 3][col] = 1
        V[b_base_j + 0][col] = 1
        W[w_base_k + 2][col] = 1
        W[w_base_k + 3][col] = -1
        col += 1
        
        # 3. M3 = A00(B10-B11)
        U[a_base_i + 0][col] = 1
        V[b_base_j + 2][col] = 1
        V[b_base_j + 3][col] = -1
        W[w_base_k + 1][col] = 1
        W[w_base_k + 3][col] = 1
        col += 1
        
        # 4. M4 = A11(B01-B00)
        U[a_base_i + 3][col] = 1
        V[b_base_j + 1][col] = 1
        V[b_base_j + 0][col] = -1
        W[w_base_k + 0][col] = 1
        W[w_base_k + 2][col] = 1
        col += 1
        
        # 5. M5 = (A00+A01)B11
        U[a_base_i + 0][col] = 1
        U[a_base_i + 1][col] = 1
        V[b_base_j + 3][col] = 1
        W[w_base_k + 0][col] = -1
        W[w_base_k + 1][col] = 1
        col += 1
        
        # 6. M6 = (A10-A00)(B00+B10)
        U[a_base_i + 2][col] = 1
        U[a_base_i + 0][col] = -1
        V[b_base_j + 0][col] = 1
        V[b_base_j + 2][col] = 1
        W[w_base_k + 0][col] = 1
        W[w_base_k + 3][col] = 1
        col += 1
        
        # 7. M7 = (A01-A11)(B10+B11)
        U[a_base_i + 1][col] = 1
        U[a_base_i + 3][col] = -1
        V[b_base_j + 2][col] = 1
        V[b_base_j + 3][col] = 1
        W[w_base_k + 0][col] = 1
        W[w_base_k + 1][col] = -1
        col += 1
        
    # Helper to add a 2x2x1 Schoolbook block
    def add_schoolbook_221(a_base_i, a_base_j, b_base_j, b_base_k, w_base_k, w_base_i):
        nonlocal col
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        # Wait, 2x2x1 means A is 2x2, B is 2x1, C is 2x1.
        # B has 1 column.
        # B[0,0] and B[1,0].
        # V indices: j*p+k. k is fixed.
        
        # Term 1: A00 B00 -> C00
        U[a_base_i + 0][col] = 1
        V[b_base_j + 0][col] = 1
        W[w_base_k + 0][col] = 1
        col += 1
        
        # Term 2: A01 B10 -> C00
        U[a_base_i + 1][col] = 1
        V[b_base_j + 2][col] = 1 # j=1, k=0 -> idx 1*p+0. If p=5, idx 5?
        # Wait, V is (m*p, rank). m=4, p=5.
        # In the local 2x2x1 block, B is 2x1.
        # The global B index j is offset by b_base_j.
        # The global B index k is offset by b_base_k.
        # V index = (b_base_j + j_local) * p + (b_base_k + k_local).
        
        # Let's refine the helper to take global offsets and local dims.
        
    # Re-implementation with correct global indexing
    
    # Block 1: A_L (rows 0-1, cols 0-1 of A) * B_T (rows 0-1, cols 0-4 of B)
    # A_L is 2x2. B_T is 2x5.
    # Split B_T into B_T1 (2x2) and B_T2 (2x3).
    
    # A_L base: i in 0..1, j in 0..1.
    # B_T base: j in 0..1, k in 0..4.
    # C_T base: k in 0..4, i in 0..1.
    
    # B_T1: k in 0..1.
    add_strassen_222(0, 0, 0, 0, 0, 0)
    
    # B_T2: k in 2..4.
    # Split into 2x1 (k=2) and 2x2 (k=3,4).
    
    # 2x1 part (k=2)
    # A_L (2x2) * B_col2 (2x1)
    # U: A_L. V: B_col2. W: C_col2.
    # A_L indices: 0,1,2,3.
    # B_col2 indices: j=0,k=2 -> V[2]; j=1,k=2 -> V[7]. (p=5)
    # C_col2 indices: k=2,i=0 -> W[2]; k=2,i=1 -> W[7]. (n=2)
    
    # Term 1: A00 B00 -> C00
    U[0][col] = 1
    V[0*5 + 2][col] = 1
    W[2*2 + 0][col] = 1
    col += 1
    
    # Term 2: A01 B10 -> C00
    U[1][col] = 1
    V[1*5 + 2][col] = 1
    W[2*2 + 0][col] = 1
    col += 1
    
    # Term 3: A10 B00 -> C10
    U[2][col] = 1
    V[0*5 + 2][col] = 1
    W[2*2 + 1][col] = 1
    col += 1
    
    # Term 4: A11 B10 -> C10
    U[3][col] = 1
    V[1*5 + 2][col] = 1
    W[2*2 + 1][col] = 1
    col += 1
    
    # 2x2 part (k=3,4)
    # A_L (2x2) * B_sub (2x2)
    # B_sub j in 0..1, k in 3..4.
    # V indices: j=0,k=3 -> 3; j=0,k=4 -> 4; j=1,k=3 -> 8; j=1,k=4 -> 9.
    # W indices: k=3,i=0 -> 6; k=3,i=1 -> 7; k=4,i=0 -> 8; k=4,i=1 -> 9.
    add_strassen_222(0, 0, 0, 3, 6, 0) # a_base=0, b_base_j=0, b_base_k=3, w_base_k=6, w_base_i=0
    
    # Block 2: A_R (rows 0-1, cols 2-3 of A) * B_B (rows 2-3, cols 0-4 of B)
    # A_R is 2x2. B_B is 2x5.
    # A_R indices: 4,5,6,7.
    # B_B indices: j=2,3. k=0..4.
    # V indices: j=2,k=0 -> 10; ...
    # C_B indices: k=0..4, i=0..1.
    # W indices: k=0,i=0 -> 0; ... (Same W as C_T? No, C is 2x5. C_T is top half? No.)
    # C = A_L B_T + A_R B_B.
    # C_T and C_B are the SAME C matrix.
    # So W indices are shared.
    
    # B_B1: k in 0..1.
    # A_R base: i in 0..1, j in 2..3.
    # A_R indices: 4,5,6,7.
    # B_B1 indices: j=2,3; k=0,1.
    # V indices: 10,11,15,16.
    # W indices: 0,1,2,3.
    
    # Modify add_strassen_222 to accept bases
    # It currently assumes 0-based local.
    # I will just call it with correct offsets.
    
    # A_R: a_base_i=0, a_base_j=2.
    # B_B1: b_base_j=2, b_base_k=0.
    # W: w_base_k=0, w_base_i=0.
    
    # The helper add_strassen_222 uses:
    # U[a_base_i + 0] ...
    # So for A_R, a_base_i should be 0, but we need to add 2 to the column index?
    # No, U index is i*m+j.
    # A_R is rows 0,1 and cols 2,3.
    # A[0,2] is idx 2. A[0,3] is idx 3.
    # A[1,2] is idx 6. A[1,3] is idx 7.
    # My helper assumes A[0,0], A[0,1], A[1,0], A[1,1] are contiguous.
    # They are NOT contiguous in U for A_R.
    # U[0,2]=idx 2, U[0,3]=idx 3.
    # U[1,2]=idx 6, U[1,3]=idx 7.
    # So I cannot use the simple helper for A_R without modification.
    
    # I will write a generic add_strassen that takes explicit index lists.
    
    # Reset col
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_generic(U_idx, V_idx, W_idx):
        nonlocal col
        # U_idx: [u00, u01, u10, u11]
        # V_idx: [v00, v01, v10, v11]
        # W_idx: [w00, w01, w10, w11]
        
        # M1
        U[U_idx[0]][col] += 1
        U[U_idx[3]][col] += 1
        V[V_idx[0]][col] += 1
        V[V_idx[3]][col] += 1
        W[W_idx[0]][col] += 1
        W[W_idx[3]][col] += 1
        W[W_idx[1]][col] += 1
        W[W_idx[2]][col] += 1
        col += 1
        
        # M2
        U[U_idx[2]][col] += 1
        U[U_idx[3]][col] += 1
        V[V_idx[0]][col] += 1
        W[W_idx[2]][col] += 1
        W[W_idx[3]][col] -= 1
        col += 1
        
        # M3
        U[U_idx[0]][col] += 1
        V[V_idx[2]][col] += 1
        V[V_idx[3]][col] -= 1
        W[W_idx[1]][col] += 1
        W[W_idx[3]][col] += 1
        col += 1
        
        # M4
        U[U_idx[3]][col] += 1
        V[V_idx[1]][col] += 1
        V[V_idx[0]][col] -= 1
        W[W_idx[0]][col] += 1
        W[W_idx[2]][col] += 1
        col += 1
        
        # M5
        U[U_idx[0]][col] += 1
        U[U_idx[1]][col] += 1
        V[V_idx[3]][col] += 1
        W[W_idx[0]][col] -= 1
        W[W_idx[1]][col] += 1
        col += 1
        
        # M6
        U[U_idx[2]][col] += 1
        U[U_idx[0]][col] -= 1
        V[V_idx[0]][col] += 1
        V[V_idx[2]][col] += 1
        W[W_idx[0]][col] += 1
        W[W_idx[3]][col] += 1
        col += 1
        
        # M7
        U[U_idx[1]][col] += 1
        U[U_idx[3]][col] -= 1
        V[V_idx[2]][col] += 1
        V[V_idx[3]][col] += 1
        W[W_idx[0]][col] += 1
        W[W_idx[1]][col] -= 1
        col += 1

    def add_schoolbook_generic(U_idx, V_idx, W_idx):
        nonlocal col
        # U_idx: [u00, u01, u10, u11]
        # V_idx: [v00, v10] (2x1)
        # W_idx: [w00, w10] (2x1)
        
        # A00 B00 -> C00
        U[U_idx[0]][col] += 1
        V[V_idx[0]][col] += 1
        W[W_idx[0]][col] += 1
        col += 1
        
        # A01 B10 -> C00
        U[U_idx[1]][col] += 1
        V[V_idx[1]][col] += 1
        W[W_idx[0]][col] += 1
        col += 1
        
        # A10 B00 -> C10
        U[U_idx[2]][col] += 1
        V[V_idx[0]][col] += 1
        W[W_idx[1]][col] += 1
        col += 1
        
        # A11 B10 -> C10
        U[U_idx[3]][col] += 1
        V[V_idx[1]][col] += 1
        W[W_idx[1]][col] += 1
        col += 1

    # Indices for A_L (2x2)
    # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    U_AL = [0, 1, 2, 3]
    
    # Indices for A_R (2x2)
    # A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    U_AR = [2, 3, 6, 7]
    
    # Block 1: A_L * B_T
    # B_T (2x5) split into B_T1 (2x2, k=0,1) and B_T2 (2x3, k=2,3,4)
    
    # B_T1 (k=0,1)
    # V[0,0]=0, V[0,1]=1, V[1,0]=5, V[1,1]=6
    V_BT1 = [0, 1, 5, 6]
    # W (k=0,1)
    # W[0,0]=0, W[0,1]=1, W[1,0]=2, W[1,1]=3
    W_CT1 = [0, 1, 2, 3]
    
    add_strassen_generic(U_AL, V_BT1, W_CT1)
    
    # B_T2 (k=2,3,4)
    # Split into B_T2a (k=2) and B_T2b (k=3,4)
    
    # B_T2a (k=2)
    # V[0,2]=2, V[1,2]=7
    V_BT2a = [2, 7]
    # W (k=2)
    # W[2,0]=4, W[2,1]=5
    W_CT2a = [4, 5]
    
    add_schoolbook_generic(U_AL, V_BT2a, W_CT2a)
    
    # B_T2b (k=3,4)
    # V[0,3]=3, V[0,4]=4, V[1,3]=8, V[1,4]=9
    V_BT2b = [3, 4, 8, 9]
    # W (k=3,4)
    # W[3,0]=6, W[3,1]=7, W[4,0]=8, W[4,1]=9
    W_CT2b = [6, 7, 8, 9]
    
    add_strassen_generic(U_AL, V_BT2b, W_CT2b)
    
    # Block 2: A_R * B_B
    # B_B (2x5) split into B_B1 (2x2, k=0,1) and B_B2 (2x3, k=2,3,4)
    
    # B_B1 (k=0,1)
    # V[2,0]=10, V[2,1]=11, V[3,0]=15, V[3,1]=16
    V_BB1 = [10, 11, 15, 16]
    # W (k=0,1) - Same as W_CT1
    W_CB1 = [0, 1, 2, 3]
    
    add_strassen_generic(U_AR, V_BB1, W_CB1)
    
    # B_B2 (k=2,3,4)
    # Split into B_B2a (k=2) and B_B2b (k=3,4)
    
    # B_B2a (k=2)
    # V[2,2]=12, V[3,2]=17
    V_BB2a = [12, 17]
    # W (k=2) - Same as W_CT2a
    W_CB2a = [4, 5]
    
    add_schoolbook_generic(U_AR, V_BB2a, W_CB2a)
    
    # B_B2b (k=3,4)
    # V[2,3]=13, V[2,4]=14, V[3,3]=18, V[3,4]=19
    V_BB2b = [13, 14, 18, 19]
    # W (k=3,4) - Same as W_CT2b
    W_CB2b = [6, 7, 8, 9]
    
    add_strassen_generic(U_AR, V_BB2b, W_CB2b)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
